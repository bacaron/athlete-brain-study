import os,sys
import requests
import json

def product_json_tractprofile_grabber(topPath,projectID,datatypeID,tag):
	
	import requests
	import os
	import json

	#load the jwt token (run bl login to create this file)
	jwt_file = open(os.environ['HOME']+'/.config/brainlife.io/.jwt', mode='r')
	jwt = jwt_file.read()

	#query datasets records
	find = { 
	    'datatype': datatypeID, 
	    'project': projectID,
	    'tags': tag }
	params = { 
	    'limit': 100, 
	    'select': 'product meta', #we just want product and meta
	    'find': json.JSONEncoder().encode(find) }
	res = requests.get('https://brainlife.io/api/warehouse/dataset/', params=params, headers={'Authorization': 'Bearer '+jwt})
	if res.status_code != 200:
	    raise Error("failed to download datasets list")

	#load product.json for each dataset
	data = res.json()
	for dataset in data["datasets"]:
	    id = dataset["_id"]
	    subject = dataset["meta"]["subject"]
	    print('<div style="float: left; height: 250px; width: 250px;">');
	    print("<b>"+subject+"</b><br>")

	    res = requests.get('https://brainlife.io/api/warehouse/dataset/product/'+dataset["_id"], headers={'Authorization': 'Bearer '+jwt})
	    if res.status_code != 200:
	        raise Error("failed to download product.json")
	    data = res.json()
	    product = data["product"]


if __name__ == '__main__':
	product_json_tractprofile_grabber(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])