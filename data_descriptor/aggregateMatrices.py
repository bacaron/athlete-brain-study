#!/usr/bin/env python

import os,sys
import csv
import json
import numpy

def aggregateMatrices(topDir,config_path):
	import os,sys
	import csv
	import json
	import numpy

	plots = []

	# load measures and compute std/mean
	with open(config_path) as config_f:
		config = json.load(config_f)

		densities = []
		counts = []
		# lengths = []
		#denlens = []
		fas = []
		# denfas = []
		# lengths = []
		# tors = []

		for output_dir in config["outputs"]:
			# density
			print("loading", output_dir+"/density.csv")
			density = numpy.genfromtxt(os.path.join(topDir+output_dir+"/networks/output/density.csv"), delimiter=',')
			densities.append(density)

			# counts
			print("loading", output_dir+"/count.csv")
			count = numpy.genfromtxt(os.path.join(topDir+output_dir+"/networks/output/count.csv"), delimiter=',')
			counts.append(count)

			# # curvs
			# print("loading", output_dir+"/curv.csv")
			# curv = numpy.genfromtxt(topDir+output_dir+"/snr/curv.csv", delimiter=',')
			# curvs.append(curv)

			# # denlens	
			# print("loading", output_dir+"/denlen.csv")
			# denlen = numpy.genfromtxt(topDir+output_dir+"/snr/denlen.csv", delimiter=',')
			# denlens.append(denlen)

			# fa
			print("loading", output_dir+"/fa_mean.csv")
			fa = numpy.genfromtxt(os.path.join(topDir+output_dir+"/networks/output/fa_mean.csv"), delimiter=',')
			fas.append(fa)

			# # lengths
			# print("loading", output_dir+"/length.csv")
			# length = numpy.genfromtxt(topDir+output_dir+"/snr/length.csv", delimiter=',')
			# lengths.append(length)

			# # tors
			# print("loading", output_dir+"/tors.csv")
			# tor = numpy.genfromtxt(topDir+output_dir+"/snr/tors.csv", delimiter=',')
			# tors.append(tor)

		# compute means and standard deviation of each measure
		density_std = numpy.std(densities, axis=0)
		numpy.savetxt('density.std.csv', density_std)
		density_mean = numpy.mean(densities, axis=0)
		numpy.savetxt('density.mean.csv', density_mean)

		count_std = numpy.std(counts, axis=0)
		numpy.savetxt('count.std.csv', count_std)
		count_mean = numpy.mean(counts, axis=0)
		numpy.savetxt('count.mean.csv', count_mean)

		# curv_std = numpy.std(curvs, axis=0)
		# numpy.savetxt('curv.std.csv', curv_std)
		# curv_mean = numpy.mean(curvs, axis=0)
		# numpy.savetxt('curv.mean.csv', curv_mean)

		# denlen_std = numpy.std(denlens, axis=0)
		# numpy.savetxt('denlen.std.csv', denlen_std)
		# denlen_mean = numpy.mean(denlens, axis=0)
		# numpy.savetxt('denlen.mean.csv', denlen_mean)

		fa_std = numpy.std(fas, axis=0)
		numpy.savetxt('fa.std.csv', fa_std)
		fa_mean = numpy.mean(fas, axis=0)
		numpy.savetxt('fa.mean.csv', fa_mean)

		# length_std = numpy.std(lengths, axis=0)
		# numpy.savetxt('length.std.csv', count_std)
		# length_mean = numpy.mean(lengths, axis=0)
		# numpy.savetxt('length.mean.csv', count_mean)

		# tor_std = numpy.std(tors, axis=0)
		# numpy.savetxt('tor.std.csv', tor_std)
		# tor_mean = numpy.mean(tors, axis=0)
		# numpy.savetxt('tor.mean.csv', tor_mean)

	layout = {
	    "xaxis": {
	        "constrain": "domain"
	    },
	    "yaxis": {
	        "autorange": "reversed",
	        "scaleanchor": "x",
	    },

	}

	#generate heatmap from std/mean
	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "density std"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Portland",
	    "z": density_std.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "density mean"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Hot",
	    "z": density_mean.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "count std"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Portland",
	    "z": count_std.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "count mean"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Hot",
	    "z": count_mean.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "curv std"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Portland",
	#     "z": curv_std.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "curv mean"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Hot",
	#     "z": curv_mean.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "denlen std"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Portland",
	#     "z": denlen_std.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "denlen mean"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Hot",
	#     "z": denlen_mean.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "fa std"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Portland",
	    "z": fa_std.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	plot = {}
	plot["type"] = "plotly"
	plot["name"] = "fa mean"
	plot["data"] = [{
	    "type": "heatmap",
	    "colorscale": "Hot",
	    "z": fa_mean.tolist(),
	}]
	plot["layout"] = layout
	plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "length std"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Portland",
	#     "z": length_std.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "length mean"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Hot",
	#     "z": length_mean.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "tors std"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Portland",
	#     "z": tor_std.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	# plot = {}
	# plot["type"] = "plotly"
	# plot["name"] = "tors mean"
	# plot["data"] = [{
	#     "type": "heatmap",
	#     "colorscale": "Hot",
	#     "z": tor_mean.tolist(),
	# }]
	# plot["layout"] = layout
	# plots.append(plot)

	#save product.json
	product = {}
	product["brainlife"] = plots
	with open("product.json", "w") as fp:
	    json.dump(product, fp)

if __name__ == '__main__':
	aggregateMatrices(sys.argv[1],sys.argv[2])