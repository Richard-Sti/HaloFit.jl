using ArgParse

include("fit_script.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--kind"
        help = "Kind of simulation"
    end

    return parse_args(s)
end

args = parse_commandline()

if args["kind"] == "csiborg1"
    fit_csiborg1()
elseif occursin("csiborg2", args["kind"])
    fit_csiborg2(args["kind"])
else
    error("Unknown simulation kind: `$(args["kind"])`")
end
