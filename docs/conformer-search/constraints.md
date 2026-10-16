# Conformer Search Constraints

Conformer Search combines molecular-graph rules with soft numerical penalties. These mechanisms restrict a search, but they are not a force field and do not prove that a candidate is chemically stable.

## Connectivity and rotatable bonds

The tool first builds a graph from interatomic distances and covalent radii. The default cutoff is 1.30 times the sum of the two covalent radii. Automatic metal connectivity is deliberately conservative:

- recognized N, O, F, P, S, Cl, Br, I, As, and Se contacts may be retained as metal–donor bonds;
- metal–carbon and metal–metal contacts are skipped by default; and
- other close metal contacts that do not pass the donor policy appear under **⚠ Skipped contacts**.

In **② Rotatable bonds**, use the contextual Ribbon's **Remove Bond** to mark a detected connection as false and **Add Bond** to supply missing connectivity. Manual removal overrides automatic detection, and manual addition can restore a contact rejected by the automatic policy. Rebuilding connectivity also rebuilds the rotatable-bond list and lock state.

A detected bond is offered as rotatable only when it joins two non-hydrogen atoms, is not a ring bond, has a neighbour on each side, and does not include the metal. Metal-mediated paths are excluded from ring detection. This is a topology rule, not a bond-order or electronic-structure analysis.

## Fixed atoms and bond locking

The metal is always fixed implicitly. User-selected atoms in **① Fixed atoms** influence which dihedrals remain available and add a numerical constraint in the objective.

With **Anchor mode (recommended for tetrapodal/tripodal ligands)** enabled, a fixed atom directly at the rotating-side anchor can remain the pivot while the arm beyond it rotates. A fixed atom deeper in the rotating subtree locks the bond. With Anchor mode disabled, any fixed atom in the rotating subtree, including the directly bonded anchor, locks the bond. The latter is stricter and can leave no free dihedrals.

The local **Filter element** selector and contextual Ribbon **Fix All Shown**, **Unfix All**, and **Invert** actions only change which atoms are marked. Use **Ribbon → Apply Locks** to recalculate and inspect the lock classification before running.

## Position mode

For **Constraint model: position**, the objective adds a large quadratic penalty for Cartesian displacement of fixed atoms from their input coordinates. Bond locking is the primary protection; the positional term is a soft numerical backstop.

Position mode is appropriate when the selected atoms should remain at the same XYZ positions. It does not restrain every unselected atom, perform geometry optimization, or enforce experimental distance restraints.

## Angular mode

For **Constraint model: angular**, each fixed atom is represented by its unit direction from the metal. The objective penalizes changes in those directions while allowing the metal–atom radii to change. It also penalizes non-hydrogen atoms within the coordination search region when their metal distance falls below an element-dependent lower bound derived from covalent radii and donor offsets.

Angular mode therefore preserves selected metal-centred directions, not exact Cartesian positions or exact metal–donor distances. Radial changes are reported as diagnostics. Only atoms marked fixed contribute a directional constraint; the short-distance coordination screen is broader and applies to nearby non-hydrogen atoms.

## Rigid-body anchor constraints

In **Rigid-body anchor rotation**, fixed atoms other than the metal are used as anchors. **Axis mode** defines the metal-centred rotation axis:

| Axis mode | Axis construction |
| --- | --- |
| `first` | Metal-to-first-anchor direction. |
| `bisector` | Normalized sum of the metal-to-anchor unit directions; with one anchor it reduces to `first`. |
| `connecting` | First-anchor to second-anchor direction, shifted to pass through the metal; with fewer than two anchors it falls back to `first`. |

**Lock mode** determines which coordinates can change:

| Lock mode | Preserved behavior |
| --- | --- |
| `rigid_with_axis` | All non-metal atoms, including the anchors, rotate together as one rigid body around the axis. |
| `angular_only` | Anchor Cartesian coordinates remain fixed; all other non-metal atoms rotate around the anchor-defined axis. |
| `angular_radial_free` | Anchor directions from the metal remain fixed, their radii are fitted within **Radial bounds rel.**, and the other non-metal atoms rotate. |

The radial bounds are multiplicative factors of each initial metal–anchor distance. The rigid-body objective also includes clash and short metal-contact penalties. The dihedral **Constraint model** selection is not used by this branch.

## Clash and geometry penalties

The clash term is a smooth quadratic penalty for close non-bonded atom pairs based on van der Waals radii. Bonded and graph-near pairs are excluded from the normal clash screen. A final integer clash count is reported with each candidate.

The clash and coordination terms are heuristic search penalties. They do not include electronic energy, solvent, bond stretching energy, conformational entropy, or a complete treatment of coordination chemistry. A zero clash count is therefore a useful screen, not evidence that the geometry is a physical minimum.

See [Optimization](optimization.md) for how these penalties combine with the PCS residual objective.
