# Methodology Asset Manifest

This manifest covers every figure still included by the owned manuscript/deck
methodology and literature files. Hashes are SHA-256. Both outputs are generated
by `figs/generate_methodology_assets.sh` (SHA-256
`38664903948da025a03921dbc1af6856d3dabdfabf8d94e54b925ad36a813b33`)
with `rsvg-convert` 2.62.3.

| Included asset | Content SHA-256 | Traceable source/configuration |
|---|---|---|
| `figs/sota_venn.png` | `bbc8e69f7f5452edcfd355efaa96bc38f6e8f8c6a8abc59b9186fb8329bab32c` | `figs/sota_venn.svg`, SHA-256 `77e73abc1a4743d16cf9426be05f5de4445739a31b02e4dc40b9c062359343a4`; generator command `rsvg-convert --width=2400 --output=sota_venn.png sota_venn.svg`; no external configuration file |
| `figs/conops.png` | `bed231b6a8c43c1a188307cff486dd3a540db70b6eef0fc22e49b2d87acd4ad2` | `figs/conops_redraw.svg`, SHA-256 `0507e9bda9293f557db3056a70b00b37c59df074eb202be5850449e24c3084f1`; generator command `rsvg-convert --background-color=white --width=2621 --output=conops.png conops_redraw.svg`; no external configuration file |

Untraceable methodology assets were removed from use, including the former
force-model plots, split-policy outcome image, encounter-plane diagnostics,
transfer/dust/optimization pipeline redraws, and the untracked defense-deck
literature gap image. Their files remain untouched unless separately owned by
another artifact; they are not part of the active methodology evidence chain.
