module HaloFit

# Include necessary files
include("main.jl")

# Exported functions and values
export Halo,
       periodic_distance!,
       center_of_mass!,
       shrinking_sphere_cm!,
       spherical_overdensity_mass,
       angular_momentum,
       λbullock,
       nfw_concentration,
       reduced_inertia_tensor,
       ellipsoid_axes_ratio,
       ρcrit0

end # module HaloFit
