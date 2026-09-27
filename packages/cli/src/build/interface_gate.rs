//! The interface gate decides if an edit to a workspace crate can change the code that a
//! dependent crate compiles.
//!
//! A dependent crate embeds these parts of a crate: every signature and type, and the body of
//! every function that rustc exports as MIR. rustc exports the body of a generic function, of an
//! `#[inline]` function, of a `const fn`, of an `async fn`, and of a trait default method. The
//! body of any other function stays in its own crate, and a dependent only calls it by symbol.
//!
//! The gate removes the body of each such opaque function from the old and the new syntax tree,
//! removes the doc attributes and the `#[cfg(test)]` items, and compares the rest. If the two
//! trees are equal, the edit is a body-only edit, and no dependent needs a rebuild.
//!
//! The gate is conservative. It keeps a body when it cannot prove that the body is opaque: an
//! unknown attribute can be a proc macro, and a nested `impl` block is visible to trait
//! resolution in every crate.
//!
//! A private function whose body the gate keeps can still be local to its crate. rustc
//! instantiates a generic function, and exports the body of an `#[inline]` or `async` function,
//! only for the code that calls it. When every caller of a private function is an opaque body,
//! no dependent compiles its body. The gate finds these functions in each file: the function
//! has no visibility modifier, its module has no child module in another file, and no kept part
//! of the file names it. The search repeats for the kept functions that a kept part names,
//! because a dependent compiles their bodies too. The gate then treats the body as opaque. The
//! search compares names only, so a method or a function in another scope with the same name
//! also counts as a use. A module in another file sees the private items of its parent
//! modules. When the caller gives the names of the kept parts of those files, see
//! `names_in_child_modules`, the rule also applies to the top level of a file with such modules.
//!
//! A private `use` declaration at the top of the file is not an interface by itself. The gate
//! compares the names that the declarations bind. A binding that changes only matters when the
//! kept part of the file uses that name. A `pub use`, a glob import and a `use` under a `cfg`
//! attribute always compare exactly.
//!
//! An edit that only adds items is an additive change. An existing dependent cannot name a new
//! free function, a new type or a new inherent method, so the dependents keep their code. The
//! gate reports the additive change apart from an interface change, and the builder replays
//! the crate alone. A dependent that then uses the new item fails to compile against the old
//! metadata, and the builder retries that patch with the cascade. The gate still reports an
//! interface change for a new trait, a new trait impl, a new macro, a new module, a new
//! re-export, a new enum variant, a new struct field, and a new inherent method whose name is a
//! method name of a common standard trait, because each of those can change the code that an
//! existing dependent compiles.
//!
//! Known limit: the automatic cross-crate inlining of small functions at `opt-level >= 1` exports
//! bodies that the gate treats as opaque. The build must pass
//! `-Zcross-crate-inline-threshold=never`.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    Attribute, Block, GenericParam, Generics, ImplItem, Item, ItemUse, Signature, Type, UseTree,
    Visibility,
};

/// The result of the gate for one edited file.
#[derive(Debug, PartialEq, Eq)]
pub enum InterfaceChange {
    /// Every difference sits in the body of an opaque function, in a doc attribute or in a
    /// `#[cfg(test)]` item.
    Same,
    /// The new file adds items that no existing dependent can name. The string lists them.
    Additive(String),
    /// The string describes the first item whose interface differs.
    Changed(String),
}

/// Returns `true` when `old` and `new` have the same interface.
pub fn same_interface(old: &syn::File, new: &syn::File) -> bool {
    interface_change(old, new) == InterfaceChange::Same
}

/// Compares the interface of `old` and `new`.
pub fn interface_change(old: &syn::File, new: &syn::File) -> InterfaceChange {
    interface_change_in(old, new, None)
}

/// Compares the interface of `old` and `new`. `child_names` holds the names in the kept parts of
/// the modules in other files that the file declares, from `names_in_child_modules`.
pub fn interface_change_in(
    old: &syn::File,
    new: &syn::File,
    child_names: Option<&HashSet<String>>,
) -> InterfaceChange {
    let mut old = old.clone();
    let mut new = new.clone();
    Stripper.visit_file_mut(&mut old);
    Stripper.visit_file_mut(&mut new);
    let local: HashSet<String> = local_kept_functions(&old.items, child_names)
        .intersection(&local_kept_functions(&new.items, child_names))
        .cloned()
        .collect();
    let root_local = child_names.is_some();
    strip_local_kept_functions(&mut old.items, &local, root_local);
    strip_local_kept_functions(&mut new.items, &local, root_local);
    if old == new {
        return InterfaceChange::Same;
    }
    if old.attrs != new.attrs {
        return InterfaceChange::Changed("the inner attributes of the file".to_string());
    }

    let (old_uses, old_rest) = split_private_uses(old.items);
    let (new_uses, new_rest) = split_private_uses(new.items);
    let mut added = Vec::new();
    if let Err(reason) = added_items(&old_rest, &new_rest, &mut added) {
        return InterfaceChange::Changed(reason);
    }

    let old_bindings = UseBindings::collect(&old_uses);
    let new_bindings = UseBindings::collect(&new_uses);
    if old_bindings.globs != new_bindings.globs {
        return InterfaceChange::Changed("a glob import".to_string());
    }
    let mut used = HashSet::new();
    let mut collector = IdentCollector(&mut used);
    for item in old_rest.iter().chain(&new_rest) {
        collector.visit_item(item);
    }
    let names: BTreeSet<&String> = old_bindings
        .names
        .keys()
        .chain(new_bindings.names.keys())
        .collect();
    for name in names {
        if old_bindings.names.get(name) != new_bindings.names.get(name) && used.contains(name) {
            return InterfaceChange::Changed(format!("the import of {name}"));
        }
    }
    if added.is_empty() {
        InterfaceChange::Same
    } else {
        InterfaceChange::Additive(added.join(", "))
    }
}

/// Walks `old` and `new` in step. An item of `new` that has no equal item in `old` must be
/// additive, and every item of `old` must appear in `new`. The walk descends into an inherent
/// `impl` block and into an inline module with the same header. The result lists the added
/// items in `added`, or describes the first difference that is not additive.
fn added_items(old: &[Item], new: &[Item], added: &mut Vec<String>) -> Result<(), String> {
    let mut i = 0;
    let mut j = 0;
    while j < new.len() {
        if i < old.len() && old[i] == new[j] {
            i += 1;
            j += 1;
            continue;
        }
        if i < old.len() {
            match (&old[i], &new[j]) {
                (Item::Impl(old_impl), Item::Impl(new_impl))
                    if same_inherent_impl_header(old_impl, new_impl) =>
                {
                    added_methods(old_impl, new_impl, added)
                        .map_err(|_| describe_item_change(&old[i], &new[j]))?;
                    i += 1;
                    j += 1;
                    continue;
                }
                (Item::Mod(old_mod), Item::Mod(new_mod))
                    if old_mod.ident == new_mod.ident
                        && old_mod.attrs == new_mod.attrs
                        && old_mod.vis == new_mod.vis =>
                {
                    if let (Some((_, old_content)), Some((_, new_content))) =
                        (&old_mod.content, &new_mod.content)
                    {
                        added_items(old_content, new_content, added)?;
                        i += 1;
                        j += 1;
                        continue;
                    }
                }
                _ => {}
            }
            // An item with the same kind and name in both files changed. It is not new, and
            // the message must name the change.
            if describe_item(&old[i]) == describe_item(&new[j]) {
                return Err(describe_item_change(&old[i], &new[j]));
            }
        }
        match additive_item(&new[j]) {
            Some(name) => {
                added.push(name);
                j += 1;
            }
            None if i < old.len() => return Err(describe_item_change(&old[i], &new[j])),
            None => return Err(format!("{} is new", describe_item(&new[j]))),
        }
    }
    if i < old.len() {
        return Err(format!("{} is not in the new file", describe_item(&old[i])));
    }
    Ok(())
}

/// Returns the description of `item` when an existing dependent cannot name it: a new function,
/// a new type, a new constant, or an inherent `impl` block whose items are all additive.
fn additive_item(item: &Item) -> Option<String> {
    match item {
        Item::Fn(_)
        | Item::Struct(_)
        | Item::Enum(_)
        | Item::Union(_)
        | Item::Type(_)
        | Item::Const(_)
        | Item::Static(_) => Some(describe_item(item)),
        Item::Impl(block) if block.trait_.is_none() => {
            let mut added = Vec::new();
            added_methods(
                &syn::ItemImpl {
                    items: Vec::new(),
                    ..block.clone()
                },
                block,
                &mut added,
            )
            .ok()?;
            Some(describe_item(item))
        }
        _ => None,
    }
}

fn same_inherent_impl_header(old: &syn::ItemImpl, new: &syn::ItemImpl) -> bool {
    old.trait_.is_none()
        && new.trait_.is_none()
        && old.attrs == new.attrs
        && old.generics == new.generics
        && old.self_ty == new.self_ty
        && old.unsafety == new.unsafety
        && old.defaultness == new.defaultness
}

/// Walks the items of two inherent `impl` blocks with the same header, like `added_items`. A
/// new method or a new associated constant is additive, unless the method name is the name of
/// a method of a common standard trait, because an inherent method hides the trait method in
/// every caller.
fn added_methods(
    old: &syn::ItemImpl,
    new: &syn::ItemImpl,
    added: &mut Vec<String>,
) -> Result<(), ()> {
    let name = describe_type(&new.self_ty);
    let mut i = 0;
    let mut j = 0;
    while j < new.items.len() {
        if i < old.items.len() && old.items[i] == new.items[j] {
            i += 1;
            j += 1;
            continue;
        }
        match &new.items[j] {
            ImplItem::Fn(method) if !hides_a_trait_method(&method.sig) => {
                added.push(format!("method {name}::{}", method.sig.ident));
            }
            ImplItem::Const(constant) => {
                added.push(format!("associated const {name}::{}", constant.ident));
            }
            _ => return Err(()),
        }
        j += 1;
    }
    if i < old.items.len() {
        return Err(());
    }
    Ok(())
}

/// The method names of the standard traits that a dependent calls on a value of any type. An
/// inherent method with one of these names takes the call from the trait method.
const TRAIT_METHOD_NAMES: &[&str] = &[
    "add",
    "as_mut",
    "as_ref",
    "borrow",
    "borrow_mut",
    "clamp",
    "clone",
    "clone_from",
    "cmp",
    "default",
    "deref",
    "deref_mut",
    "deserialize",
    "div",
    "drop",
    "eq",
    "extend",
    "flush",
    "fmt",
    "from",
    "from_iter",
    "from_str",
    "ge",
    "gt",
    "hash",
    "index",
    "index_mut",
    "into",
    "into_iter",
    "le",
    "lt",
    "max",
    "min",
    "mul",
    "ne",
    "neg",
    "next",
    "not",
    "partial_cmp",
    "poll",
    "product",
    "read",
    "rem",
    "serialize",
    "source",
    "sub",
    "sum",
    "to_owned",
    "to_string",
    "try_from",
    "try_into",
    "write",
    "write_fmt",
    "write_str",
];

fn hides_a_trait_method(sig: &Signature) -> bool {
    TRAIT_METHOD_NAMES.contains(&sig.ident.to_string().as_str())
}

/// Separates the private, unconditional `use` declarations from the other items.
fn split_private_uses(items: Vec<Item>) -> (Vec<ItemUse>, Vec<Item>) {
    let mut uses = Vec::new();
    let mut rest = Vec::new();
    for item in items {
        match item {
            Item::Use(use_item)
                if matches!(use_item.vis, Visibility::Inherited)
                    && use_item
                        .attrs
                        .iter()
                        .all(|attr| !attr.path().is_ident("cfg")) =>
            {
                uses.push(use_item)
            }
            other => rest.push(other),
        }
    }
    (uses, rest)
}

/// The names that a set of `use` declarations binds, each with the path that it binds to.
#[derive(Default)]
struct UseBindings {
    names: BTreeMap<String, String>,
    globs: BTreeSet<String>,
}

impl UseBindings {
    fn collect(uses: &[ItemUse]) -> Self {
        let mut bindings = Self::default();
        for use_item in uses {
            let prefix = if use_item.leading_colon.is_some() {
                "::"
            } else {
                ""
            };
            bindings.visit_tree(&use_item.tree, prefix);
        }
        bindings
    }

    fn visit_tree(&mut self, tree: &UseTree, prefix: &str) {
        match tree {
            UseTree::Path(path) => {
                let prefix = format!("{prefix}{}::", path.ident);
                self.visit_tree(&path.tree, &prefix);
            }
            UseTree::Name(name) => {
                if name.ident == "self" {
                    let path = prefix.trim_end_matches("::");
                    let last = path.rsplit("::").next().unwrap_or(path);
                    self.names.insert(last.to_string(), path.to_string());
                } else {
                    self.names
                        .insert(name.ident.to_string(), format!("{prefix}{}", name.ident));
                }
            }
            UseTree::Rename(rename) => {
                self.names.insert(
                    rename.rename.to_string(),
                    format!("{prefix}{}", rename.ident),
                );
            }
            UseTree::Glob(_) => {
                self.globs.insert(prefix.to_string());
            }
            UseTree::Group(group) => {
                for item in &group.items {
                    self.visit_tree(item, prefix);
                }
            }
        }
    }
}

/// Collects every identifier in the visited items, the tokens of macro invocations and of
/// attribute lists included.
struct IdentCollector<'a>(&'a mut HashSet<String>);

impl<'a, 'ast> Visit<'ast> for IdentCollector<'a> {
    fn visit_ident(&mut self, ident: &'ast proc_macro2::Ident) {
        self.0.insert(ident.to_string());
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        syn::visit::visit_macro(self, mac);
        self.collect_tokens(mac.tokens.clone());
    }

    fn visit_meta_list(&mut self, list: &'ast syn::MetaList) {
        syn::visit::visit_meta_list(self, list);
        self.collect_tokens(list.tokens.clone());
    }
}

impl IdentCollector<'_> {
    fn collect_tokens(&mut self, tokens: TokenStream) {
        for tree in tokens {
            match tree {
                TokenTree::Ident(ident) => {
                    self.0.insert(ident.to_string());
                }
                TokenTree::Group(group) => self.collect_tokens(group.stream()),
                TokenTree::Punct(_) | TokenTree::Literal(_) => {}
            }
        }
    }
}

/// A private function that `local_kept_functions` can treat as local: the gate keeps its body
/// only because it is generic, a method of a generic impl, `async`, `#[inline]` or has
/// `impl Trait` in its signature. A `const fn` can run in the type of a public item, and an
/// unknown attribute or an item in the body can reach other crates, so they do not qualify.
fn is_local_candidate(
    attrs: &[Attribute],
    vis: &Visibility,
    sig: &Signature,
    impl_generics: Option<&Generics>,
    body: &Block,
) -> bool {
    matches!(vis, Visibility::Inherited)
        && sig.constness.is_none()
        && !body_has_visible_item(body)
        && attrs
            .iter()
            .all(|attr| is_opaque_attribute(attr) || attr.path().is_ident("inline"))
        && keep_reason(attrs, sig, impl_generics, body).is_some()
}

/// True when `items` or an inline module in them declares a module in another file. The
/// private items of `items` are visible in that file.
fn has_module_in_other_file(items: &[Item]) -> bool {
    items.iter().any(|item| match item {
        Item::Mod(module) => match &module.content {
            None => true,
            Some((_, items)) => has_module_in_other_file(items),
        },
        _ => false,
    })
}

/// The names of the private kept functions of `items` that no dependent can compile. See the
/// module documentation. The caller runs the `Stripper` on `items` first, so the bodies of the
/// opaque functions are empty and do not count as a use.
/// `child_names` holds the names in the modules in other files that the top level of `items`
/// declares. Without it, the top level of a file with such modules has no local function.
fn local_kept_functions(items: &[Item], child_names: Option<&HashSet<String>>) -> HashSet<String> {
    fn scan(
        items: &[Item],
        root_local: bool,
        candidates: &mut Vec<(String, HashSet<String>)>,
        used: &mut HashSet<String>,
    ) {
        let local_module = root_local || !has_module_in_other_file(items);
        let body_names = |body: &Block| {
            let mut names = HashSet::new();
            IdentCollector(&mut names).visit_block(body);
            names
        };
        for item in items {
            match item {
                Item::Fn(function)
                    if local_module
                        && is_local_candidate(
                            &function.attrs,
                            &function.vis,
                            &function.sig,
                            None,
                            &function.block,
                        ) =>
                {
                    candidates.push((function.sig.ident.to_string(), body_names(&function.block)));
                }
                Item::Impl(block) if block.trait_.is_none() => {
                    let mut collector = IdentCollector(used);
                    for attr in &block.attrs {
                        collector.visit_attribute(attr);
                    }
                    collector.visit_generics(&block.generics);
                    collector.visit_type(&block.self_ty);
                    for impl_item in &block.items {
                        match impl_item {
                            ImplItem::Fn(method)
                                if local_module
                                    && is_local_candidate(
                                        &method.attrs,
                                        &method.vis,
                                        &method.sig,
                                        Some(&block.generics),
                                        &method.block,
                                    ) =>
                            {
                                candidates.push((
                                    method.sig.ident.to_string(),
                                    body_names(&method.block),
                                ));
                            }
                            other => IdentCollector(used).visit_impl_item(other),
                        }
                    }
                }
                Item::Mod(module) if module.content.is_some() => {
                    let mut collector = IdentCollector(used);
                    for attr in &module.attrs {
                        collector.visit_attribute(attr);
                    }
                    let (_, items) = module.content.as_ref().unwrap();
                    scan(items, false, candidates, used);
                }
                other => IdentCollector(used).visit_item(other),
            }
        }
    }

    let mut candidates = Vec::new();
    let mut used = child_names.cloned().unwrap_or_default();
    scan(items, child_names.is_some(), &mut candidates, &mut used);
    // A kept function that a kept part names is compiled by a dependent too, so the names in
    // its body count as a use. Repeat until no new function is found.
    let mut exposed: HashSet<String> = HashSet::new();
    loop {
        let mut found = false;
        for (name, _) in &candidates {
            if used.contains(name) && exposed.insert(name.clone()) {
                found = true;
            }
        }
        if !found {
            break;
        }
        for (name, names) in &candidates {
            if exposed.contains(name) {
                used.extend(names.iter().cloned());
            }
        }
    }
    candidates
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| !exposed.contains(name))
        .collect()
}

/// Empty the body of each private kept function in `items` whose name is in `names`. The
/// selection repeats the one of `local_kept_functions`.
fn strip_local_kept_functions(items: &mut [Item], names: &HashSet<String>, root_local: bool) {
    if names.is_empty() {
        return;
    }
    let local_module = root_local || !has_module_in_other_file(items);
    for item in items {
        match item {
            Item::Fn(function)
                if local_module
                    && names.contains(&function.sig.ident.to_string())
                    && is_local_candidate(
                        &function.attrs,
                        &function.vis,
                        &function.sig,
                        None,
                        &function.block,
                    ) =>
            {
                function.block = Box::new(empty_block());
            }
            Item::Impl(block) if block.trait_.is_none() && local_module => {
                let generics = block.generics.clone();
                for impl_item in &mut block.items {
                    if let ImplItem::Fn(method) = impl_item {
                        if names.contains(&method.sig.ident.to_string())
                            && is_local_candidate(
                                &method.attrs,
                                &method.vis,
                                &method.sig,
                                Some(&generics),
                                &method.block,
                            )
                        {
                            method.block = empty_block();
                        }
                    }
                }
            }
            Item::Mod(module) => {
                if let Some((_, items)) = &mut module.content {
                    strip_local_kept_functions(items, names, false);
                }
            }
            _ => {}
        }
    }
}

/// The names in the kept parts of the modules in other files that `file` declares, and of their
/// own child modules, read from the disk. `path` is the path of `file`. The result is `None` when
/// a module path is not certain: a `#[path]` attribute, a module in another file inside an inline
/// module, or a file that dx cannot read or parse.
pub fn names_in_child_modules(path: &std::path::Path, file: &syn::File) -> Option<HashSet<String>> {
    // The directory of the child modules: the directory of `lib.rs`, `main.rs` and `mod.rs`,
    // else a directory with the name of the file.
    let dir = match path.file_stem()?.to_str()? {
        "lib" | "main" | "mod" => path.parent()?.to_path_buf(),
        stem => path.parent()?.join(stem),
    };
    let mut names = HashSet::new();
    for item in &file.items {
        match item {
            Item::Mod(module) if module.content.is_none() => {
                if module.attrs.iter().any(|attr| attr.path().is_ident("path")) {
                    return None;
                }
                if is_cfg_test(Some(&module.attrs)) {
                    continue;
                }
                let name = module.ident.to_string();
                let child = [
                    dir.join(format!("{name}.rs")),
                    dir.join(&name).join("mod.rs"),
                ]
                .into_iter()
                .find(|candidate| candidate.exists())?;
                let mut child_file =
                    syn::parse_file(&std::fs::read_to_string(&child).ok()?).ok()?;
                Stripper.visit_file_mut(&mut child_file);
                IdentCollector(&mut names).visit_file(&child_file);
                names.extend(names_in_child_modules(&child, &child_file)?);
            }
            Item::Mod(module) => {
                let (_, items) = module.content.as_ref()?;
                if has_module_in_other_file(items) {
                    return None;
                }
            }
            _ => {}
        }
    }
    Some(names)
}

/// Describes the first difference between two versions of one item. For an `impl` block or a
/// function, the description names the method and says why the gate kept its body.
fn describe_item_change(old: &Item, new: &Item) -> String {
    let name = describe_item(new);
    match (old, new) {
        (Item::Fn(old_fn), Item::Fn(new_fn)) if old_fn.sig == new_fn.sig => {
            match keep_reason(&new_fn.attrs, &new_fn.sig, None, &new_fn.block) {
                Some(reason) => format!("{name}, whose body is kept because it is {reason}"),
                None => name,
            }
        }
        (Item::Impl(old_impl), Item::Impl(new_impl)) => {
            if old_impl.items.len() != new_impl.items.len() {
                return format!(
                    "{name}, with {} items before and {} after",
                    old_impl.items.len(),
                    new_impl.items.len()
                );
            }
            for (old_item, new_item) in old_impl.items.iter().zip(&new_impl.items) {
                if old_item == new_item {
                    continue;
                }
                return match (old_item, new_item) {
                    (ImplItem::Fn(old_fn), ImplItem::Fn(new_fn)) if old_fn.sig == new_fn.sig => {
                        let method = new_fn.sig.ident.to_string();
                        match keep_reason(
                            &new_fn.attrs,
                            &new_fn.sig,
                            Some(&new_impl.generics),
                            &new_fn.block,
                        ) {
                            Some(reason) => format!(
                                "{name}, method {method}, whose body is kept because it is {reason}"
                            ),
                            None => format!("{name}, method {method}"),
                        }
                    }
                    (_, ImplItem::Fn(new_fn)) => {
                        format!("{name}, the signature of method {}", new_fn.sig.ident)
                    }
                    _ => format!("{name}, an item that is not a method"),
                };
            }
            name
        }
        _ => name,
    }
}

fn describe_item(item: &Item) -> String {
    match item {
        Item::Const(i) => format!("const {}", i.ident),
        Item::Enum(i) => format!("enum {}", i.ident),
        Item::ExternCrate(i) => format!("extern crate {}", i.ident),
        Item::Fn(i) => format!("fn {}", i.sig.ident),
        Item::ForeignMod(_) => "an extern block".to_string(),
        Item::Impl(i) => match &i.trait_ {
            Some((_, path, _)) => format!(
                "impl {} for {}",
                path.segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default(),
                describe_type(&i.self_ty)
            ),
            None => format!("impl {}", describe_type(&i.self_ty)),
        },
        Item::Macro(i) => format!(
            "macro {}",
            i.mac
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default()
        ),
        Item::Mod(i) => format!("mod {}", i.ident),
        Item::Static(i) => format!("static {}", i.ident),
        Item::Struct(i) => format!("struct {}", i.ident),
        Item::Trait(i) => format!("trait {}", i.ident),
        Item::TraitAlias(i) => format!("trait alias {}", i.ident),
        Item::Type(i) => format!("type {}", i.ident),
        Item::Union(i) => format!("union {}", i.ident),
        Item::Use(_) => "a use declaration".to_string(),
        _ => "an item".to_string(),
    }
}

fn describe_type(ty: &Type) -> String {
    match ty {
        Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default(),
        _ => "a type".to_string(),
    }
}

/// Attributes that do not export a function body and do not expand to other items.
const OPAQUE_ATTRIBUTES: &[&str] = &[
    "doc",
    "allow",
    "expect",
    "warn",
    "deny",
    "forbid",
    "cfg",
    "must_use",
    "cold",
    "track_caller",
    "no_mangle",
    "deprecated",
];

struct Stripper;

impl VisitMut for Stripper {
    fn visit_file_mut(&mut self, file: &mut syn::File) {
        strip_docs(&mut file.attrs);
        file.items.retain(|item| !is_cfg_test(item_attrs(item)));
        syn::visit_mut::visit_file_mut(self, file);
    }

    fn visit_item_mod_mut(&mut self, module: &mut syn::ItemMod) {
        if let Some((_, items)) = &mut module.content {
            items.retain(|item| !is_cfg_test(item_attrs(item)));
        }
        syn::visit_mut::visit_item_mod_mut(self, module);
    }

    fn visit_item_mut(&mut self, item: &mut Item) {
        if let Some(attrs) = item_attrs_mut(item) {
            strip_docs(attrs);
        }
        if let Item::Fn(function) = item {
            if is_opaque(&function.attrs, &function.sig, None, &function.block) {
                function.block = Box::new(empty_block());
            }
        }
        syn::visit_mut::visit_item_mut(self, item);
    }

    fn visit_item_impl_mut(&mut self, block: &mut syn::ItemImpl) {
        block
            .items
            .retain(|item| !is_cfg_test(impl_item_attrs(item)));
        for item in &mut block.items {
            if let ImplItem::Fn(method) = item {
                strip_docs(&mut method.attrs);
                if is_opaque(
                    &method.attrs,
                    &method.sig,
                    Some(&block.generics),
                    &method.block,
                ) {
                    method.block = empty_block();
                }
            }
        }
        syn::visit_mut::visit_item_impl_mut(self, block);
    }

    fn visit_impl_item_mut(&mut self, item: &mut ImplItem) {
        if let Some(attrs) = impl_item_attrs_mut(item) {
            strip_docs(attrs);
        }
        syn::visit_mut::visit_impl_item_mut(self, item);
    }

    fn visit_trait_item_mut(&mut self, item: &mut syn::TraitItem) {
        if let Some(attrs) = trait_item_attrs_mut(item) {
            strip_docs(attrs);
        }
        syn::visit_mut::visit_trait_item_mut(self, item);
    }

    fn visit_field_mut(&mut self, field: &mut syn::Field) {
        strip_docs(&mut field.attrs);
        syn::visit_mut::visit_field_mut(self, field);
    }

    fn visit_variant_mut(&mut self, variant: &mut syn::Variant) {
        strip_docs(&mut variant.attrs);
        syn::visit_mut::visit_variant_mut(self, variant);
    }
}

/// A function is opaque when a dependent crate can only call it by symbol.
fn is_opaque(
    attrs: &[Attribute],
    sig: &Signature,
    impl_generics: Option<&Generics>,
    body: &Block,
) -> bool {
    keep_reason(attrs, sig, impl_generics, body).is_none()
}

/// Why the gate keeps the body of a function in the comparison, or `None` when the body is
/// opaque.
fn keep_reason(
    attrs: &[Attribute],
    sig: &Signature,
    impl_generics: Option<&Generics>,
    body: &Block,
) -> Option<String> {
    if sig.constness.is_some() {
        return Some("a const fn".to_string());
    }
    if sig.asyncness.is_some() {
        return Some("an async fn".to_string());
    }
    if has_type_or_const_params(&sig.generics) {
        return Some("a generic fn".to_string());
    }
    if impl_generics.is_some_and(has_type_or_const_params) {
        return Some("a method of a generic impl".to_string());
    }
    if let Some(attr) = attrs.iter().find(|attr| !is_opaque_attribute(attr)) {
        let name = attr
            .path()
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        return Some(format!("the attribute #[{name}]"));
    }
    if signature_has_impl_trait(sig) {
        return Some("impl Trait in the signature".to_string());
    }
    if body_has_visible_item(body) {
        return Some("an impl or a macro item in the body".to_string());
    }
    None
}

fn has_type_or_const_params(generics: &Generics) -> bool {
    generics
        .params
        .iter()
        .any(|param| !matches!(param, GenericParam::Lifetime(_)))
}

fn is_opaque_attribute(attr: &Attribute) -> bool {
    attr.path()
        .get_ident()
        .is_some_and(|ident| OPAQUE_ATTRIBUTES.contains(&ident.to_string().as_str()))
}

fn signature_has_impl_trait(sig: &Signature) -> bool {
    struct Finder(bool);
    impl<'ast> Visit<'ast> for Finder {
        fn visit_type(&mut self, ty: &'ast Type) {
            if matches!(ty, Type::ImplTrait(_)) {
                self.0 = true;
            }
            syn::visit::visit_type(self, ty);
        }
    }
    let mut finder = Finder(false);
    finder.visit_signature(sig);
    finder.0
}

/// An `impl` block inside a body is visible to trait resolution in every crate. A macro item
/// inside a body can expand to one, and so can a statement macro with braces. A `macro_rules!`
/// definition cannot, and only its name is visible outside the body, when it has
/// `#[macro_export]`.
fn body_has_visible_item(body: &Block) -> bool {
    struct Finder(bool);
    impl<'ast> Visit<'ast> for Finder {
        fn visit_item(&mut self, item: &'ast Item) {
            let visible = match item {
                Item::Impl(_) => true,
                Item::Macro(item) => {
                    !item.mac.path.is_ident("macro_rules")
                        || item
                            .attrs
                            .iter()
                            .any(|attr| attr.path().is_ident("macro_export"))
                }
                _ => false,
            };
            if visible {
                self.0 = true;
            }
            syn::visit::visit_item(self, item);
        }
        fn visit_stmt_macro(&mut self, stmt: &'ast syn::StmtMacro) {
            if matches!(stmt.mac.delimiter, syn::MacroDelimiter::Brace(_)) {
                self.0 = true;
            }
            syn::visit::visit_stmt_macro(self, stmt);
        }
    }
    let mut finder = Finder(false);
    finder.visit_block(body);
    finder.0
}

fn empty_block() -> Block {
    syn::parse_quote!({})
}

fn strip_docs(attrs: &mut Vec<Attribute>) {
    attrs.retain(|attr| !attr.path().is_ident("doc"));
}

fn is_cfg_test(attrs: Option<&[Attribute]>) -> bool {
    attrs.is_some_and(|attrs| {
        attrs.iter().any(|attr| {
            if !attr.path().is_ident("cfg") {
                return false;
            }
            let mut is_test = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("test") {
                    is_test = true;
                }
                Ok(())
            });
            is_test
        })
    })
}

fn item_attrs(item: &Item) -> Option<&[Attribute]> {
    Some(match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => return None,
    })
}

fn item_attrs_mut(item: &mut Item) -> Option<&mut Vec<Attribute>> {
    Some(match item {
        Item::Const(i) => &mut i.attrs,
        Item::Enum(i) => &mut i.attrs,
        Item::ExternCrate(i) => &mut i.attrs,
        Item::Fn(i) => &mut i.attrs,
        Item::ForeignMod(i) => &mut i.attrs,
        Item::Impl(i) => &mut i.attrs,
        Item::Macro(i) => &mut i.attrs,
        Item::Mod(i) => &mut i.attrs,
        Item::Static(i) => &mut i.attrs,
        Item::Struct(i) => &mut i.attrs,
        Item::Trait(i) => &mut i.attrs,
        Item::TraitAlias(i) => &mut i.attrs,
        Item::Type(i) => &mut i.attrs,
        Item::Union(i) => &mut i.attrs,
        Item::Use(i) => &mut i.attrs,
        _ => return None,
    })
}

fn impl_item_attrs(item: &ImplItem) -> Option<&[Attribute]> {
    Some(match item {
        ImplItem::Const(i) => &i.attrs,
        ImplItem::Fn(i) => &i.attrs,
        ImplItem::Type(i) => &i.attrs,
        ImplItem::Macro(i) => &i.attrs,
        _ => return None,
    })
}

fn impl_item_attrs_mut(item: &mut ImplItem) -> Option<&mut Vec<Attribute>> {
    Some(match item {
        ImplItem::Const(i) => &mut i.attrs,
        ImplItem::Fn(i) => &mut i.attrs,
        ImplItem::Type(i) => &mut i.attrs,
        ImplItem::Macro(i) => &mut i.attrs,
        _ => return None,
    })
}

fn trait_item_attrs_mut(item: &mut syn::TraitItem) -> Option<&mut Vec<Attribute>> {
    use syn::TraitItem;
    Some(match item {
        TraitItem::Const(i) => &mut i.attrs,
        TraitItem::Fn(i) => &mut i.attrs,
        TraitItem::Type(i) => &mut i.attrs,
        TraitItem::Macro(i) => &mut i.attrs,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        InterfaceChange, interface_change, interface_change_in, names_in_child_modules,
        same_interface,
    };

    fn gate(old: &str, new: &str) -> bool {
        let old = syn::parse_file(old).expect("old source parses");
        let new = syn::parse_file(new).expect("new source parses");
        same_interface(&old, &new)
    }

    /// `true` when the gate reports an additive change.
    fn additive(old: &str, new: &str) -> bool {
        let old = syn::parse_file(old).expect("old source parses");
        let new = syn::parse_file(new).expect("new source parses");
        matches!(interface_change(&old, &new), InterfaceChange::Additive(_))
    }

    #[test]
    fn new_fn_type_and_const_are_additive() {
        assert!(additive(
            "pub fn f() {}",
            "pub fn f() {} pub fn g() -> u32 { 1 }"
        ));
        assert!(additive("pub fn f() {}", "pub struct S; pub fn f() {}"));
        assert!(additive(
            "pub fn f() {}",
            "pub fn f() {} pub const N: u32 = 1;"
        ));
        assert!(additive(
            "pub fn f() {}",
            "pub fn f() {} pub fn g<T>() -> T { todo!() }"
        ));
    }

    #[test]
    fn new_inherent_method_is_additive() {
        assert!(additive(
            "pub struct S; impl S { pub fn a(&self) {} }",
            "pub struct S; impl S { pub fn a(&self) {} pub fn b(&self) {} }",
        ));
        assert!(additive(
            "pub struct S; impl S { pub fn a(&self) {} }",
            "pub struct S; impl S { pub const N: u32 = 1; pub fn a(&self) {} }",
        ));
        assert!(additive(
            "pub struct S;",
            "pub struct S; impl S { pub fn a(&self) {} }",
        ));
    }

    #[test]
    fn new_inherent_method_with_a_trait_method_name_is_an_interface_change() {
        assert!(!additive(
            "pub struct S; impl S { pub fn a(&self) {} }",
            "pub struct S; impl S { pub fn a(&self) {} pub fn clone(&self) -> S { S } }",
        ));
    }

    #[test]
    fn new_item_in_inline_module_is_additive() {
        assert!(additive(
            "pub mod m { pub fn f() {} }",
            "pub mod m { pub fn f() {} pub fn g() {} }",
        ));
    }

    #[test]
    fn additive_change_with_a_body_edit_is_additive() {
        assert!(additive(
            "pub fn f() -> u32 { 1 }",
            "pub fn g() -> u32 { 2 } pub fn f() -> u32 { g() }",
        ));
    }

    #[test]
    fn new_trait_impl_macro_module_and_re_export_are_interface_changes() {
        assert!(!additive("pub fn f() {}", "pub fn f() {} pub trait T {}"));
        assert!(!additive(
            "pub struct S;",
            "pub struct S; impl Default for S { fn default() -> S { S } }",
        ));
        assert!(!additive(
            "pub fn f() {}",
            "pub fn f() {} macro_rules! m { () => {} }"
        ));
        assert!(!additive("pub fn f() {}", "pub fn f() {} pub mod m {}"));
        assert!(!additive(
            "pub fn f() {}",
            "pub fn f() {} pub use std::cmp::max;"
        ));
        assert!(!gate("pub fn f() {}", "pub fn f() {} pub trait T {}"));
    }

    #[test]
    fn new_variant_and_new_field_are_interface_changes() {
        assert!(!additive("pub enum E { A }", "pub enum E { A, B }"));
        assert!(!additive(
            "pub struct S { a: u32 }",
            "pub struct S { a: u32, b: u32 }"
        ));
    }

    #[test]
    fn removed_or_changed_item_is_an_interface_change() {
        assert!(!additive("pub fn f() {} pub fn g() {}", "pub fn f() {}"));
        assert!(!additive("pub fn f() {}", "pub fn f(x: u32) {}"));
        assert!(!additive(
            "pub struct S; impl S { pub fn a(&self) {} }",
            "pub struct S; impl S { pub fn a(&self, x: u32) {} pub fn b(&self) {} }",
        ));
    }

    #[test]
    fn body_edit_in_plain_fn_is_body_only() {
        assert!(gate(
            "pub fn f(x: u32) -> u32 { x + 1 }",
            "pub fn f(x: u32) -> u32 { x + 2 }",
        ));
    }

    #[test]
    fn body_edit_in_method_of_plain_impl_is_body_only() {
        assert!(gate(
            "struct S; impl S { pub fn f(&self) -> u32 { 1 } }",
            "struct S; impl S { pub fn f(&self) -> u32 { 2 } }",
        ));
    }

    #[test]
    fn lifetime_params_do_not_export_a_body() {
        assert!(gate(
            "pub fn f<'a>(x: &'a str) -> &'a str { x }",
            "pub fn f<'a>(x: &'a str) -> &'a str { &x[..] }",
        ));
    }

    #[test]
    fn signature_edit_is_an_interface_change() {
        assert!(!gate(
            "pub fn f(x: u32) -> u32 { x }",
            "pub fn f(x: u64) -> u64 { x }",
        ));
    }

    #[test]
    fn generic_body_is_an_interface() {
        assert!(!gate(
            "pub fn f<T: Default>() -> T { T::default() }",
            "pub fn f<T: Default>() -> T { let t = T::default(); t }",
        ));
    }

    #[test]
    fn generic_impl_method_body_is_an_interface() {
        assert!(!gate(
            "pub struct S<T>(T); impl<T> S<T> { pub fn f(&self) -> u32 { 1 } }",
            "pub struct S<T>(T); impl<T> S<T> { pub fn f(&self) -> u32 { 2 } }",
        ));
    }

    #[test]
    fn inline_body_is_an_interface() {
        assert!(!gate(
            "#[inline] pub fn f() -> u32 { 1 }",
            "#[inline] pub fn f() -> u32 { 2 }",
        ));
    }

    #[test]
    fn const_fn_and_async_fn_bodies_are_interfaces() {
        assert!(!gate(
            "pub const fn f() -> u32 { 1 }",
            "pub const fn f() -> u32 { 2 }",
        ));
        assert!(!gate(
            "pub async fn f() -> u32 { 1 }",
            "pub async fn f() -> u32 { 2 }",
        ));
    }

    #[test]
    fn impl_trait_in_signature_keeps_the_body() {
        assert!(!gate(
            "pub fn f() -> impl Fn() -> u32 { || 1 }",
            "pub fn f() -> impl Fn() -> u32 { || 2 }",
        ));
        assert!(!gate(
            "pub fn f(x: impl Into<u32>) -> u32 { x.into() }",
            "pub fn f(x: impl Into<u32>) -> u32 { x.into() + 1 }",
        ));
    }

    #[test]
    fn private_kept_fn_with_only_opaque_callers_is_body_only() {
        let file = |n: u32| {
            format!(
                "fn helper(ui: &mut u32, add: impl FnOnce(&mut u32)) {{ *ui += {n}; add(ui) }}
                 fn generic<T: Default>() -> T {{ let _ = {n}; T::default() }}
                 async fn fetch() -> u32 {{ {n} }}
                 pub fn show(ui: &mut u32) {{ helper(ui, |_| {{}}); let _: u32 = generic(); }}
                 pub struct S;
                 impl S {{
                     fn private_method(&self, f: impl Fn() -> u32) -> u32 {{ f() + {n} }}
                     pub fn run(&self) -> u32 {{ self.private_method(|| 1) }}
                 }}"
            )
        };
        assert!(gate(&file(1), &file(2)));
    }

    #[test]
    fn private_kept_fn_that_a_kept_body_calls_is_an_interface() {
        // A public generic function calls the helper, so a dependent compiles the helper.
        assert!(!gate(
            "fn helper(x: impl Into<u32>) -> u32 { x.into() }
             pub fn show<T: Into<u32>>(x: T) -> u32 { helper(x) }",
            "fn helper(x: impl Into<u32>) -> u32 { x.into() + 1 }
             pub fn show<T: Into<u32>>(x: T) -> u32 { helper(x) }",
        ));
        // The same through a second private generic function.
        assert!(!gate(
            "fn inner<T>(_: T) -> u32 { 1 }
             fn outer<T>(x: T) -> u32 { inner(x) }
             #[inline] pub fn show() -> u32 { outer(0u8) }",
            "fn inner<T>(_: T) -> u32 { 2 }
             fn outer<T>(x: T) -> u32 { inner(x) }
             #[inline] pub fn show() -> u32 { outer(0u8) }",
        ));
        // A generic method of a trait impl calls the helper.
        assert!(!gate(
            "fn helper<T>(_: T) -> u32 { 1 }
             pub struct W<T>(T);
             impl<T: Clone> Clone for W<T> { fn clone(&self) -> Self { helper(0u8); W(self.0.clone()) } }",
            "fn helper<T>(_: T) -> u32 { 2 }
             pub struct W<T>(T);
             impl<T: Clone> Clone for W<T> { fn clone(&self) -> Self { helper(0u8); W(self.0.clone()) } }",
        ));
    }

    #[test]
    fn private_kept_fn_visible_to_another_file_is_an_interface() {
        // A child module in another file can call the helper from a public generic function.
        assert!(!gate(
            "mod child; fn helper<T>(_: T) -> u32 { 1 }",
            "mod child; fn helper<T>(_: T) -> u32 { 2 }",
        ));
        // `pub(crate)` makes the helper visible to the other files of the crate.
        assert!(!gate(
            "pub(crate) fn helper<T>(_: T) -> u32 { 1 }",
            "pub(crate) fn helper<T>(_: T) -> u32 { 2 }",
        ));
        // A child module in another file sees only the items of its own parent modules, so a
        // helper in a sibling inline module stays local.
        assert!(gate(
            "mod other; mod local { fn helper<T>(_: T) -> u32 { 1 } pub fn f() -> u32 { helper(0u8) } }",
            "mod other; mod local { fn helper<T>(_: T) -> u32 { 2 } pub fn f() -> u32 { helper(0u8) } }",
        ));
    }

    /// The gate result for `old` and `new` at `dir/lib.rs`, with the names of its child modules.
    fn gate_with_children(dir: &std::path::Path, old: &str, new: &str) -> bool {
        let old = syn::parse_file(old).expect("old source parses");
        let new = syn::parse_file(new).expect("new source parses");
        let names = names_in_child_modules(&dir.join("lib.rs"), &new);
        interface_change_in(&old, &new, names.as_ref()) == InterfaceChange::Same
    }

    #[test]
    fn private_kept_fn_with_child_module_files_uses_their_names() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quiet.rs"),
            "pub fn g() -> u32 { super::helper(0u8) }",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("loud")).unwrap();
        std::fs::write(dir.path().join("loud/mod.rs"), "mod inner;").unwrap();
        std::fs::write(
            dir.path().join("loud/inner.rs"),
            "pub fn g<T>(_: T) -> u32 { super::super::shown(0u8) }",
        )
        .unwrap();
        // The opaque body of `quiet::g` calls `helper`, so `helper` stays local.
        assert!(gate_with_children(
            dir.path(),
            "mod quiet; mod loud; fn helper<T>(_: T) -> u32 { 1 }",
            "mod quiet; mod loud; fn helper<T>(_: T) -> u32 { 2 }",
        ));
        // A generic function in a grandchild module calls `shown`, so its body is an interface.
        assert!(!gate_with_children(
            dir.path(),
            "mod quiet; mod loud; fn shown<T>(_: T) -> u32 { 1 }",
            "mod quiet; mod loud; fn shown<T>(_: T) -> u32 { 2 }",
        ));
        // dx cannot find the file of `missing`, so every body in the file stays an interface.
        assert!(!gate_with_children(
            dir.path(),
            "mod missing; fn helper<T>(_: T) -> u32 { 1 }",
            "mod missing; fn helper<T>(_: T) -> u32 { 2 }",
        ));
    }

    #[test]
    fn private_kept_fn_with_a_local_macro_rules_is_body_only() {
        assert!(gate(
            "fn helper<T>(_: T) -> u32 { macro_rules! one { () => { 1 } } one!() }",
            "fn helper<T>(_: T) -> u32 { macro_rules! one { () => { 2 } } one!() }",
        ));
        // An exported macro is visible to the dependents.
        assert!(!gate(
            "fn helper<T>(_: T) -> u32 { #[macro_export] macro_rules! one { () => { 1 } } 1 }",
            "fn helper<T>(_: T) -> u32 { #[macro_export] macro_rules! one { () => { 2 } } 1 }",
        ));
        // Another item macro can expand to an `impl` block.
        assert!(!gate(
            "fn helper<T>(_: T) -> u32 { make_impl! { A } 1 }",
            "fn helper<T>(_: T) -> u32 { make_impl! { A } 2 }",
        ));
    }

    #[test]
    fn changed_kept_fn_message_names_the_change() {
        let old = syn::parse_file("pub fn f<T>(_: T) -> u32 { 1 } pub fn g() {}").unwrap();
        let new = syn::parse_file("pub fn f<T>(_: T) -> u32 { 2 } pub fn g() {}").unwrap();
        assert_eq!(
            interface_change(&old, &new),
            InterfaceChange::Changed(
                "fn f, whose body is kept because it is a generic fn".to_string()
            )
        );
    }

    #[test]
    fn private_const_fn_and_unknown_attribute_keep_the_body() {
        assert!(!gate(
            "const fn helper() -> u32 { 1 } pub fn f() -> u32 { helper() }",
            "const fn helper() -> u32 { 2 } pub fn f() -> u32 { helper() }",
        ));
        assert!(!gate(
            "#[some_macro] fn helper<T>(_: T) -> u32 { 1 }",
            "#[some_macro] fn helper<T>(_: T) -> u32 { 2 }",
        ));
    }

    #[test]
    fn trait_default_method_body_is_an_interface() {
        assert!(!gate(
            "pub trait T { fn f(&self) -> u32 { 1 } }",
            "pub trait T { fn f(&self) -> u32 { 2 } }",
        ));
    }

    #[test]
    fn unknown_attribute_keeps_the_body() {
        assert!(!gate(
            "#[wasm_bindgen] pub fn f() -> u32 { 1 }",
            "#[wasm_bindgen] pub fn f() -> u32 { 2 }",
        ));
    }

    #[test]
    fn nested_impl_in_body_keeps_the_body() {
        // A new `impl` inside a body is visible to trait resolution in every crate.
        assert!(!gate(
            "pub fn f() { struct L; }",
            "pub fn f() { struct L; impl Default for L { fn default() -> L { L } } }",
        ));
        // The body of a plain method inside that `impl` is opaque like any other body.
        assert!(gate(
            "pub fn f() { struct L; impl Default for L { fn default() -> L { L } } }",
            "pub fn f() { struct L; impl Default for L { fn default() -> L { let l = L; l } } }",
        ));
    }

    #[test]
    fn doc_edit_is_body_only() {
        assert!(gate(
            "/// Old doc.\npub struct S { /// Field.\n pub x: u32 }",
            "/// New doc.\npub struct S { /// Other.\n pub x: u32 }",
        ));
    }

    #[test]
    fn test_module_edit_is_body_only() {
        assert!(gate(
            "pub fn f() {} #[cfg(test)] mod tests { #[test] fn t() { assert!(true); } }",
            "pub fn f() {} #[cfg(test)] mod tests { #[test] fn t() { assert!(false); } }",
        ));
    }

    #[test]
    fn new_import_for_a_body_is_body_only() {
        assert!(gate(
            "pub fn f() -> u32 { 1 }",
            "use std::cmp::max; pub fn f() -> u32 { max(1, 2) }",
        ));
        assert!(gate(
            "use a::{B, C}; pub fn f(b: B) -> u32 { 1 }",
            "use a::{B, C, D}; pub fn f(b: B) -> u32 { D::g() }",
        ));
    }

    #[test]
    fn import_of_a_name_in_a_signature_is_an_interface() {
        assert!(!gate(
            "use a::Foo; pub fn f(x: Foo) {}",
            "use b::Foo; pub fn f(x: Foo) {}",
        ));
        assert!(!gate(
            "use a::Foo; pub fn f(x: Foo) {}",
            "use b::Bar as Foo; pub fn f(x: Foo) {}",
        ));
    }

    #[test]
    fn import_of_a_name_in_a_kept_body_is_an_interface() {
        assert!(!gate(
            "use a::g; pub fn f<T>() { g::<T>() }",
            "use b::g; pub fn f<T>() { g::<T>() }",
        ));
        assert!(!gate("use a::g; m! { g }", "use b::g; m! { g }",));
        assert!(!gate(
            "use a::Tr; #[derive(Tr)] pub struct S;",
            "use b::Tr; #[derive(Tr)] pub struct S;",
        ));
    }

    #[test]
    fn re_export_and_glob_and_cfg_imports_compare_exactly() {
        assert!(!gate("pub use a::Foo;", "pub use b::Foo;"));
        assert!(!gate("pub fn f() {}", "use a::*; pub fn f() {}"));
        assert!(!gate(
            "#[cfg(feature = \"x\")] use a::Foo; pub fn f() {}",
            "#[cfg(feature = \"x\")] use b::Foo; pub fn f() {}",
        ));
    }

    #[test]
    fn self_import_binds_the_module_name() {
        assert!(!gate(
            "use a::{self}; pub fn f() -> a::T { todo!() }",
            "use b::a; pub fn f() -> a::T { todo!() }",
        ));
    }

    #[test]
    fn struct_field_edit_is_an_interface_change() {
        assert!(!gate(
            "pub struct S { x: u32 }",
            "pub struct S { x: u32, y: u32 }",
        ));
    }

    #[test]
    fn const_value_edit_is_an_interface_change() {
        assert!(!gate("pub const N: u32 = 1;", "pub const N: u32 = 2;"));
    }

    #[test]
    fn macro_rules_edit_is_an_interface_change() {
        assert!(!gate(
            "macro_rules! m { () => { 1 } }",
            "macro_rules! m { () => { 2 } }",
        ));
    }

    #[test]
    fn body_edit_in_inline_module_is_body_only() {
        assert!(gate(
            "mod inner { pub fn f() -> u32 { 1 } }",
            "mod inner { pub fn f() -> u32 { 2 } }",
        ));
    }
}
