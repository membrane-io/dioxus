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
//! A private `use` declaration at the top of the file is not an interface by itself. The gate
//! compares the names that the declarations bind. A binding that changes only matters when the
//! kept part of the file uses that name. A `pub use`, a glob import and a `use` under a `cfg`
//! attribute always compare exactly.
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

/// Returns `true` when `old` and `new` have the same interface: every difference between them
/// sits in the body of an opaque function, in a doc attribute or in a `#[cfg(test)]` item.
pub fn same_interface(old: &syn::File, new: &syn::File) -> bool {
    interface_change(old, new).is_none()
}

/// Returns a description of the first top-level item whose interface differs between `old` and
/// `new`, or `None` when the two files have the same interface.
pub fn interface_change(old: &syn::File, new: &syn::File) -> Option<String> {
    let mut old = old.clone();
    let mut new = new.clone();
    Stripper.visit_file_mut(&mut old);
    Stripper.visit_file_mut(&mut new);
    if old == new {
        return None;
    }
    if old.attrs != new.attrs {
        return Some("the inner attributes of the file".to_string());
    }

    let (old_uses, old_rest) = split_private_uses(old.items);
    let (new_uses, new_rest) = split_private_uses(new.items);
    if old_rest != new_rest {
        for (old_item, new_item) in old_rest.iter().zip(&new_rest) {
            if old_item != new_item {
                return Some(describe_item_change(old_item, new_item));
            }
        }
        return Some(format!(
            "the item count, {} before and {} after",
            old_rest.len(),
            new_rest.len()
        ));
    }

    let old_bindings = UseBindings::collect(&old_uses);
    let new_bindings = UseBindings::collect(&new_uses);
    if old_bindings.globs != new_bindings.globs {
        return Some("a glob import".to_string());
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
            return Some(format!("the import of {name}"));
        }
    }
    None
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
/// inside a body can expand to one.
fn body_has_visible_item(body: &Block) -> bool {
    struct Finder(bool);
    impl<'ast> Visit<'ast> for Finder {
        fn visit_item(&mut self, item: &'ast Item) {
            if matches!(item, Item::Impl(_) | Item::Macro(_)) {
                self.0 = true;
            }
            syn::visit::visit_item(self, item);
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
    use super::same_interface;

    fn gate(old: &str, new: &str) -> bool {
        let old = syn::parse_file(old).expect("old source parses");
        let new = syn::parse_file(new).expect("new source parses");
        same_interface(&old, &new)
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
            "struct S<T>(T); impl<T> S<T> { fn f(&self) -> u32 { 1 } }",
            "struct S<T>(T); impl<T> S<T> { fn f(&self) -> u32 { 2 } }",
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
