# Copyright (C) 2023 Richard Stiskalek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
using DataFrames, HaloFit, HDF5, ProgressMeter
import H5Zblosc


################################################################################
#                               File paths                                     #
################################################################################


begin
    zfill(n::Int, width::Int) = zfill(string(n), width)
    zfill(n::String, width::Int) = lpad(n, width, '0')
end

"""
    find_csiborg1_final_snapshot_paths(nsim::Int)

Find the path to the final snapshot and its FoF catalogue of the CSiBORG-1
simulation with the given `nsim` IC index.
"""
function find_csiborg1_final_snapshot_paths(nsim::Int)
    folder_path = "/mnt/extraspace/rstiskalek/csiborg1/chain_$nsim"
    files = readdir(folder_path)  # Read all files in the directory

    snapshot_files = filter(f -> occursin(r"^snapshot_\d{5}\.hdf5$", f) && f != "snapshot_00001.hdf5", files)
    if length(snapshot_files) == 0
        error("No snapshot file found.")
    elseif length(snapshot_files) > 1
        error("Multiple snapshot files found.")
    else
        snapshot_path = joinpath(folder_path, snapshot_files[1])
    end

    fof_files = filter(f -> startswith(f, "fof_"), files)
    if length(fof_files) == 0
        error("No FoF file found.")
    elseif length(fof_files) > 1
        error("Multiple FoF files found.")
    else
        fof_path = joinpath(folder_path, fof_files[1])
    end

    return snapshot_path, fof_path
end


"""
    find_csiborg2_final_snapshot_paths(nsim::Int, kind::String)

Find the path to the final snapshot and its FoF catalogue of the CSiBORG-2
simulation with the given `nsim` IC index.
"""
function find_csiborg2_final_snapshot_paths(nsim::Int, kind::String)
    if !(kind in ["main", "random", "varysmall"])
        error("Unknown kind: `$(kind)`")
    end

    if kind == "varysmall"
        nsim = zfill(nsim, 3)
        snapshot_path = "/mnt/extraspace/rstiskalek/csiborg2_$kind/chain_16417_$nsim/output/snapshot_099_full.hdf5"
        fof_path = "/mnt/extraspace/rstiskalek/csiborg2_$kind/chain_16417_$nsim/output/fof_subhalo_tab_099.hdf5"
    else
        snapshot_path = "/mnt/extraspace/rstiskalek/csiborg2_$kind/chain_$nsim/output/snapshot_099_full.hdf5"
        fof_path = "/mnt/extraspace/rstiskalek/csiborg2_$kind/chain_$nsim/output/fof_subhalo_tab_099.hdf5"
    end

    if !isfile(snapshot_path)
        error("File does not exist: `$(snapshot_path)`")
    end

    if !isfile(fof_path)
        error("File does not exist: `$(fof_path)`")
    end

    return snapshot_path, fof_path
end


################################################################################
#                               Halo loading                                   #
################################################################################


"""
    make_offsets_csiborg1(nsim::Int)

Make a dictionary of halo IDs to their offsets in the FoF catalogue of CSiBORG1.
"""
function make_offsets_csiborg1(nsim::Int)
    __, fof_path = find_csiborg1_final_snapshot_paths(nsim)

    offsets = nothing
    h5open(fof_path, "r") do file
        if haskey(file, "GroupOffset")
            offsets = Int64.(read(file["GroupOffset"]))
        else
            error("No `GroupOffset` dataset found.")
        end
    end

    hids = offsets[1, :]
    nhalo = length(hids)

    hid2offset= Dict{Int64, Vector{Int64}}()
    for n in 1:nhalo
        hid = offsets[1, n]
        i, j = offsets[2, n], offsets[3, n]

        hid2offset[hid] = [i + 1, j + 1]
    end

    return hid2offset
end


"""
    make_offsets_csiborg2(nsim::Int, kind::String)

Make a dictionary of halo IDs to their offsets in the FoF catalogue of CSiBORG2.
"""
function make_offsets_csiborg2(nsim::Int, kind::String)
    __, fof_path = find_csiborg2_final_snapshot_paths(nsim, kind)


    hid2offsets = Dict()
    h5open(fof_path, "r") do file
        offsets_high = file["Group/GroupOffsetType"][2, :]
        lengths_high = file["Group/GroupLenType"][2, :]
        nhalo = length(offsets_high)

        offsets_low = file["Group/GroupOffsetType"][6, :]
        lengths_low = file["Group/GroupLenType"][6, :]


        for n in 1:nhalo
            if length(lengths_high[n]) > 0
                ihigh = offsets_high[n] + 1
                jhigh = offsets_high[n] + lengths_high[n]
            else
                ihigh = 0
                jhigh = 0
            end

            ilow = offsets_low[n] + 1
            if length(lengths_low[n]) > 0
                ilow = offsets_low[n] + 1
                jlow = offsets_low[n] + lengths_low[n]
            else
                ilow = 0
                jlow = 0
            end


            hid2offsets[n - 1] = [[ihigh, jhigh], [ilow, jlow]]
        end
    end

    return hid2offsets
end


"""
    make_offsets(nsim::Int, simname::String)

Make a dictionary of halo IDs to their offsets.
"""
function make_offsets(nsim::Int, simname::String)
    if simname == "csiborg1"
        hid2offset = make_offsets_csiborg1(nsim)
    elseif occursin("csiborg2", simname)
        kind = string(split(simname, "_")[end])
        hid2offset = make_offsets_csiborg2(nsim, kind)
    else
        error("Unknown simulation name: `$(simname)`")
    end
end


"""
    load_halo_from_offsets(hid::Integer, snapshot::HDF5.File, offsets::Dict, boxsize::Real, simname::String)

Load a halo with the given `hid` from the given `snapshot` file using the given `offsets`.
"""
function load_halo_from_offsets(hid::Integer, snapshot::HDF5.File, offsets::Dict, simname::String)
    if simname == "csiborg1"
        boxsize = 677.7
        i, j = offsets[hid]
        pos = Matrix(snapshot["Coordinates"][:, i:j]')
        vel = Matrix(snapshot["Velocities"][:, i:j]')
        mass = snapshot["Masses"][i:j]
        return Halo(pos, vel, mass, boxsize)
    elseif occursin("csiborg2", simname)
        boxsize = 676.6
        i, j = offsets[hid][1]
        pos = Matrix(snapshot["PartType1/Coordinates"][:, i:j]')
        vel = Matrix(snapshot["PartType1/Velocities"][:, i:j]')
        mass = ones(Float32, size(pos, 1)) * Float32(attrs(snapshot["Header"])["MassTable"][2] * 1e10)

        ilow, jlow = offsets[hid][2]

        if ilow != 0 && jlow != 0
            pos_low = Matrix(snapshot["PartType5/Coordinates"][:, ilow:jlow]')
            vel_low = Matrix(snapshot["PartType5/Velocities"][:, ilow:jlow]')
            mass_low = snapshot["PartType5/Masses"][ilow:jlow] * Float32(1e10)

            pos = vcat(pos, pos_low)
            vel = vcat(vel, vel_low)
            mass = vcat(mass, mass_low)
        end

        return Halo(pos, vel, mass, boxsize)
    else
        error("Unknown simulation name: `$(simname)`")
    end

end


################################################################################
#                          Fitting functions                                   #
################################################################################


"""
    function fit_csiborg_final(simname::String, nsim::Int;
                               verbose::Bool=true, npart_min::Integer=100, shrink_npart_min::Int=50,
                               shrink_factor::Real=0.975)

Fit the final snapshot of the given `simname` simulation with the given `nsim`.
"""
function fit_csiborg_final(simname::String, nsim::Int;
                           verbose::Bool=true, npart_min::Integer=100, shrink_npart_min::Int=50,
                           shrink_factor::Real=0.975)
    if simname == "csiborg1"
        snapshot_path, __ = find_csiborg1_final_snapshot_paths(nsim)
    elseif occursin("csiborg2", simname)
        kind = string(split(simname, "_")[end])
        snapshot_path, __ = find_csiborg2_final_snapshot_paths(nsim, kind)
    else
        error("Unknown simulation name: `$(simname)`")
    end

    snapshot = h5open(snapshot_path, "r")
    offsets = make_offsets(nsim, simname)
    hids = sort(collect(Int64, keys(offsets)))

    ρ200c = Float32(ρcrit0(1) * 200)
    symbols = [:hid, :cmx, :cmy, :cmz, :mtot, :m200c, :r200c, :lambda200c, :conc, :q, :s, :cm_displacement]

    n_cols, n_rows = length(symbols), length(hids)

    df = DataFrame([fill(Float32(NaN), n_rows) for _ in 1:n_cols], symbols)
    p = Progress(n_rows; enabled=verbose, dt=1, barlen=50, showspeed=true)
    for i in 1:n_rows
        hid = hids[i]
        df[i, :hid] = hid

        halo = load_halo_from_offsets(hid, snapshot, offsets, simname)

        if length(halo) < npart_min
            next!(p)
            continue
        end

        df[i, :mtot] = sum(halo.mass)

        shrinking_sphere_cm!(halo; npart_min=shrink_npart_min, shrink_factor=shrink_factor)
        df[i, :cmx], df[i, :cmy], df[i, :cmz] = halo.cm[1], halo.cm[2], halo.cm[3]


        m200c, r200c = spherical_overdensity_mass(halo, ρ200c)
        df[i, :m200c], df[i, :r200c] = m200c, r200c

        if !isnan(m200c)
            angmom = angular_momentum(halo, r200c)
            df[i, :lambda200c] = λbullock(angmom, m200c, r200c)

            df[i, :conc] = nfw_concentration(halo, r200c)

            Iij = inertia_tensor(halo, r200c)
            df[i, :q], df[i, :s] = ellipsoid_axes_ratio(Iij)

            df[i, :cm_displacement] = cm_displacement(halo) / r200c
        end

        next!(p)
    end

    finish!(p)
    close(snapshot)

    return df
end


################################################################################
#                          Save processed data                                 #
################################################################################


"""

    save_frame(fout::String, df::DataFrames.DataFrame)

Save a dataframe as a HDF5 file.
"""
function save_frame(fout::String, df::DataFrames.DataFrame)
    println("Writing to ... `$(fout)`")

    cm = hcat(df.cmx, df.cmy, df.cmz)

    h5open(fout, "w") do file
        file["index"] = df[!, :hid]
        file["cm_shrink"] = Matrix(cm')

        for col in names(df)
            if col == :cmx || col == :cmy || col == :cmz
                continue
            end

            file[String(col)] = df[!, col]
        end
    end
end


################################################################################
#                               Submission                                     #
################################################################################


function fit_csiborg1(mode::String)
    # for nsim in [7444 + n * 24 for n in 0:100]
    for nsim in [7468]
        println("Fitting CSiBORG1 IC `$(nsim)`")
        res = fit_csiborg_final("csiborg1", nsim)
        fout = "/mnt/extraspace/rstiskalek/csiborg1/chain_$nsim/fitted_halos.hdf5"
        save_frame(fout, res)
    end
end


function fit_csiborg2(simname::String)
    kind = string(split(simname, "_")[end])

    if kind == "main"
        nsims = [15517, 15617, 15717, 15817, 15917, 16017, 16117, 16217, 16317,
                 16417, 16517, 16617, 16717, 16817, 16917, 17017, 17117, 17217,
                 17317, 17417]
    elseif kind == "random"
        nsims = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
                 325, 350, 375, 400, 425, 450, 475]
    elseif kind == "varysmall"
        nsims = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
                 325, 350, 375, 400, 425, 450, 475]
    else
        error("Unknown CSiBORG2 kind: `$(kind)`")
    end

    for nsim in nsims
        println("Fitting CSiBORG2_$kind IC `$(nsim)`")
        res = fit_csiborg_final(simname, nsim)
        fout = "/mnt/extraspace/rstiskalek/csiborg2_$kind/catalogues/fitted_halos_$nsim.hdf5"
        save_frame(fout, res)
    end
end


# function fit_tng300dark()
#     mpart = 0.0047271638660809 * 1e10   # Msun/h
#     boxsize = 205.0                     # Mpc/h
#
#     println("Fitting TNG300-1-Dark")
#     res = fit_from_offsets(path_tng300dark_particles(), boxsize, "tng300dark";
#                            zero_index=true, npart_min=100,
#                            verbose=true, mpart=mpart,
#                            shrink_npart_min=250, shrink_factor=0.95)
#
#     fout = "/mnt/extraspace/rstiskalek/TNG300-1-Dark/fitted_halos.hdf5"
#     save_frame(fout, res)
# end
