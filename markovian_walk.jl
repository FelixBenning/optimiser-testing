### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4a24e9bd-2a51-47ec-8220-390e7c992b84
begin
	using LinearAlgebra: LinearAlgebra, eigen, I
	using PlutoUI: Slider
	using Plots: plot
end

# ╔═╡ 84229f44-b757-40e0-a813-10cee5e721f7
using StaticArrays

# ╔═╡ 5e806095-4df2-467f-9c47-0c299acaba96
@bind states Slider(2:100, show_value=true, default=10)

# ╔═╡ 5b7f30f4-0608-4e57-97dd-0518a5e502f5
@bind epsilon Slider(0:0.01:0.99, default=0.1, show_value=true)

# ╔═╡ 5c952fde-a944-472e-a9f9-28a9d563622d
detTransitions = vcat(
	[zeros(states-1) LinearAlgebra.diagm(ones(states-1))],
	[zeros(states-1)' 1]
)

# ╔═╡ e2569eb2-2f6b-11ed-21a4-238908507381
transitions = epsilon/states * ones((states,states)) + (1-epsilon)*detTransitions

# ╔═╡ 8a913aed-724f-48b0-b4a8-2f781dc8f7e0
decomp  = eigen(transitions')

# ╔═╡ e497568b-be03-43cb-9200-14ec21d45b31
abs.(decomp.values)

# ╔═╡ 09d393c8-18d6-4ac4-9e89-0c0325f04d1b
d=sum(abs.(decomp.vectors[:,states]))

# ╔═╡ bc36c7ac-d23e-494e-b01a-5d7b9075653e
plot(1:states, abs.(decomp.vectors[:,states])/d)

# ╔═╡ 1d393cba-f5d3-4ece-a4ec-d37d389fd2d9
struct Discretization{T, dim} <: AbstractArray{T, dim}
	length::SVector{dim, Int}
	start::SVector{dim, T}
	stop::SVector{dim, T}
end

# ╔═╡ e8072a65-dfc9-4ff5-b6fb-5b157d9913f4
function Base.size(D::Discretization)
	return Tuple(D.length)
end

# ╔═╡ 71535b61-fb48-479c-a3bb-ee8b242b0d86
function Base.getindex(D::Discretization{T, dim}, keys::Vararg{Int, dim}) where {dim, T}
	for (len, i) in zip(D.length, keys)
		(1 <= i && i <= len) || throw(BoundsError(D, keys))
	end
	Tuple(Iterators.map(enumerate(keys)) do (idx, key)
		(D.stop[idx]-D.start[idx])/D.length[idx] * (key-1) + D.start[idx]
	end)
end

# ╔═╡ 7ee7dd00-ce34-4fca-9fe6-04963a4f0f75
function centeredCube(dim::Int, side::T, n::Int) where T<:Number
	return Discretization{T, dim}(
		fill(round(Int, exp(log(n)/dim)), dim),
		fill(-side/2, dim),
		fill(side/2, dim)
	)
end

# ╔═╡ 97ea19c8-1bd4-44a6-aa29-92fecdf8833d
x= centeredCube(2, 5., 1000)

# ╔═╡ 1d3b3df3-2e5b-4db4-806c-0ab8118811e9
f(x...) = LinearAlgebra.norm(x)

# ╔═╡ 21c4aca8-3f80-42da-9812-64aca218d1aa
plot(map(first,x), map(last,x), map(f, x))

# ╔═╡ 0a5cca26-6334-4a73-a0e3-b27f60a0ddad
y = map(f,x)

# ╔═╡ bd6f9853-232b-40ec-b069-0ca05c07f96d
unitvec(i, dim) = ntuple(j-> i==j, dim) |> CartesianIndex

# ╔═╡ 4951d124-22c2-406f-a64e-73c8d0e99d64
a = [
	1 2 3 1
	5 0 1 3
	2 4 4 4
]

# ╔═╡ 458114ef-a5e7-4d58-9296-21d84a97a964
struct Walk{T} <: AbstractArray{T, 2}
	jumpTo::Vector{T}
end

# ╔═╡ a30bbff6-8445-4180-8e78-7c11053ae222
function Base.size(W::Walk)
	return length(W.jumpTo), length(W.jumpTo)
end

# ╔═╡ c8c01255-86db-4ddb-82ec-9268f39fd480
function smallestNeighbour(valueGrid::AbstractArray{T,N}) where {T,N}
	indices = CartesianIndices(valueGrid)
	goto = collect(indices)
	sizes = size(valueGrid)
	for dim in 1:N
		# try to go down
		map(selectdim(indices, dim, 1:(sizes[dim]-1))) do idx
			smallest = valueGrid[goto[idx]]
			shifted = idx + unitvec(dim, length(sizes))
			if valueGrid[shifted] < smallest
				goto[idx] = shifted
			end
		end
		# try to go up
		map(selectdim(indices, dim, 2:sizes[dim])) do idx
			smallest = valueGrid[goto[idx]]
			shifted = idx - unitvec(dim, N)
			if valueGrid[shifted] < smallest
				goto[idx] = shifted
			end
		end
	end
	return goto
end

# ╔═╡ c9d2f1ac-31eb-464e-836d-facd7ba132a1
map(x->a[x], smallestNeighbour(a))

# ╔═╡ 8e3933a4-5b71-4a7a-9bea-1faa17e30174
function Base.getindex(W::Walk, i,j)
	return Int(W.jumpTo[i] == j)
end

# ╔═╡ 559cd092-6e3d-4c41-8360-a8c6b1f95e68
md"# Discrete Gradient Flow"

# ╔═╡ a034e3d1-638b-4a68-8783-13c3c19270f1
vcat(1,zeros(3))

# ╔═╡ 54e3e06e-b042-42cb-9651-f58ad8cc8418
w= reshape(map(x->LinearIndices(a)[x], smallestNeighbour(a)), :)

# ╔═╡ 3c05acf5-5c89-40d8-b579-a6678e640038
mu =  (transpose(Walk(w)) + I)\zeros(length(w))

# ╔═╡ 6fc5456b-833e-4a9a-9dfd-75b6da586eec
transpose(Walk(w)) * mu

# ╔═╡ 66eb1691-4bc1-4622-979b-9b3ab7f29e89
smallestNeighbour(a)

# ╔═╡ e3cd21fc-e440-424d-b8e6-68717e2edf73
unitvec(2, 10)

# ╔═╡ Cell order:
# ╠═4a24e9bd-2a51-47ec-8220-390e7c992b84
# ╠═5e806095-4df2-467f-9c47-0c299acaba96
# ╠═5b7f30f4-0608-4e57-97dd-0518a5e502f5
# ╠═bc36c7ac-d23e-494e-b01a-5d7b9075653e
# ╟─5c952fde-a944-472e-a9f9-28a9d563622d
# ╠═e2569eb2-2f6b-11ed-21a4-238908507381
# ╟─8a913aed-724f-48b0-b4a8-2f781dc8f7e0
# ╠═e497568b-be03-43cb-9200-14ec21d45b31
# ╠═09d393c8-18d6-4ac4-9e89-0c0325f04d1b
# ╟─559cd092-6e3d-4c41-8360-a8c6b1f95e68
# ╠═84229f44-b757-40e0-a813-10cee5e721f7
# ╠═1d393cba-f5d3-4ece-a4ec-d37d389fd2d9
# ╠═e8072a65-dfc9-4ff5-b6fb-5b157d9913f4
# ╠═71535b61-fb48-479c-a3bb-ee8b242b0d86
# ╠═7ee7dd00-ce34-4fca-9fe6-04963a4f0f75
# ╠═97ea19c8-1bd4-44a6-aa29-92fecdf8833d
# ╠═1d3b3df3-2e5b-4db4-806c-0ab8118811e9
# ╠═21c4aca8-3f80-42da-9812-64aca218d1aa
# ╠═0a5cca26-6334-4a73-a0e3-b27f60a0ddad
# ╠═c8c01255-86db-4ddb-82ec-9268f39fd480
# ╠═bd6f9853-232b-40ec-b069-0ca05c07f96d
# ╠═4951d124-22c2-406f-a64e-73c8d0e99d64
# ╠═c9d2f1ac-31eb-464e-836d-facd7ba132a1
# ╠═458114ef-a5e7-4d58-9296-21d84a97a964
# ╠═a30bbff6-8445-4180-8e78-7c11053ae222
# ╠═8e3933a4-5b71-4a7a-9bea-1faa17e30174
# ╠═3c05acf5-5c89-40d8-b579-a6678e640038
# ╠═6fc5456b-833e-4a9a-9dfd-75b6da586eec
# ╠═a034e3d1-638b-4a68-8783-13c3c19270f1
# ╠═54e3e06e-b042-42cb-9651-f58ad8cc8418
# ╠═66eb1691-4bc1-4622-979b-9b3ab7f29e89
# ╠═e3cd21fc-e440-424d-b8e6-68717e2edf73
