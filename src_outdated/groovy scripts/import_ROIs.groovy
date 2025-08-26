// Build GeoJSON filename
def entry = getProjectEntry()
def name = entry.getImageName()
println(name)
def fileName = name -".tif" + "_rois.geojson"

// Build filepath
def pathInput = buildFilePath(PROJECT_BASE_DIR, "GeoJSON", fileName)

// Import annotations
importObjectsFromFile(pathInput)

// Finished
println("Done!")