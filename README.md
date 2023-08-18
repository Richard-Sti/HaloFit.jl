# HaloFit.jl

**HaloFit.jl** is a highly-optimized Julia package offering a suite of utilities designed for analyzing dark matter halo properties within a periodic box.

## Functions
- `shrink_sphere_cm!`
- `spherical_overdensity_mass`
- `angular_momentum`
- `λbullock`
- `nfw_concentration`
- `inertia_tensor`
- `ellipsoid_axes_ratio`
- `ρcrit0`
- `cm_displacement`
- `virial_fraction`


Throughout the code, the following unit convenvention is assumed:
- Length: $\mathrm{Mpc} / h$
- Mass: $\mathrm{M}_\odot / h$
- Velocity: $\mathrm{km} / \mathrm{s}$
- Gravitational potential: $(\mathrm{km} / \mathrm{s})^2$

## Installation

To install `HaloFit.jl`, you can clone the repository directly:

```bash
git clone https://github.com/Richard-Sti/HaloFit.jl
cd HaloFit.jl
```

Start Julia and activate the project:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate(".")
Pkg.precompile(".")
```

## Dependencies

This project relies on the following Julia packages:

- [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

## Basic Usage

Here's a quick start guide on using `HaloFit.jl`:

```julia
using HaloFit

# Matrices of halo's particle positions, velocities, and masses
pos = ...
vel = ...
mass = ...

boxsize = ...

h = Halo()

shrinking_sphere_cm!(h)
@show h.cm
```

For a working example, see ...

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) - see the [LICENSE](LICENSE) file for details.
