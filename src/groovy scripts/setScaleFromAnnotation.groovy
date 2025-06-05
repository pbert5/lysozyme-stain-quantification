// ===================================================================
// setScaleFromAnnotation_LongestDimension.groovy
// -------------------------------------------------------------------
// 1) Finds the first annotation (or detection) whose class is exactly
//    "Scale".
// 2) Retrieves all contour points (Point2) from that ROI.
// 3) Computes the maximum Euclidean distance between any two points.
// 4) Prompts the user for the real‐world length (µm) of that scale bar.
// 5) Sets the pixel calibration (µm/px) accordingly.
// ===================================================================

//import qupath.lib.gui.scripting.QPEx               // Use QPEx for GUI calls
import qupath.lib.objects.PathObject
import qupath.lib.geom.Point2                        // QuPath’s Point2 class
import static qupath.lib.scripting.QP.*
import static qupath.lib.gui.scripting.QPEx.*

// 1) Get the current image data (exit if nothing open)
def imageData = getCurrentImageData()
if (imageData == null) {
    print("❌ No image is currently open or selected.")
    return
}

// 2) Search for any PathObject whose class name is "Scale"
def hierarchy   = imageData.getHierarchy()
def allObjects  = hierarchy.getFlattenedObjectList(null)
def scaleObjs   = allObjects.findAll { PathObject obj ->
    obj.getPathClass() != null && obj.getPathClass().getName().equalsIgnoreCase("Scale")
}

if (scaleObjs.isEmpty()) {
    print("❌ No object classified as “Scale” was found.")
    // Run classifier “scale” at 1× (native pixel resolution), threshold = 0
    createAnnotationsFromPixelClassifier("scale", 50.0, 0.0)

}

// 3) Take the first “Scale” object
def scaleObj = scaleObjs[0]
def roi      = scaleObj.getROI()

// 4) Retrieve all contour points (List<Point2>) from that ROI
def contourPoints = roi.getAllPoints()  // returns List<Point2>
if (contourPoints == null || contourPoints.isEmpty()) {
    print("❌ Unable to retrieve contour points for the Scale ROI.")
    return
}

// 5) Compute the maximum Euclidean distance (in pixels) between any two Point2s
double maxDistPx = 0.0

// We’ll iterate with two nested loops; untyped “it” ensures we accept Point2 objects:
for (p1 in contourPoints) {
    for (p2 in contourPoints) {
        // p1 and p2 are Point2; use getX(), getY():
        double dx = p1.getX() - p2.getX()
        double dy = p1.getY() - p2.getY()
        double d  = Math.hypot(dx, dy)
        if (d > maxDistPx) {
            maxDistPx = d
        }
    }
}

if (maxDistPx <= 0) {
    print("❌ Computed longest dimension is zero (ROI may be degenerate).")
    return
}

print("▶ ‘Scale’ object: longest dimension = " +
            String.format("%.2f", maxDistPx) + " px")

// 6) Prompt the user for the real‐world length (in microns) of the scale bar
runOnUiThread {
def answer = Dialogs.showInputDialog(
    "Scale Calibration",
    "Enter the real‐world length of this Scale bar in microns:",
    "100.0"
)
if (answer == null) {
    print("❌ Calibration canceled (no input).")
    return
}}

double realWorldMicrons
try {
    realWorldMicrons = Double.parseDouble(answer.trim())
} catch (Exception e) {
    print("❌ Invalid number: “${answer}”. Please enter a numeric value.")
    return
}
if (realWorldMicrons <= 0) {
    print("❌ The real‐world length must be greater than 0.")
    return
}

// 7) Compute microns per pixel
double pixelSize = realWorldMicrons / maxDistPx
print("▶ Setting pixel size to " +
            String.format("%.4f", pixelSize) + " µm/px " +
            "(= " + realWorldMicrons + " µm ÷ " +
            String.format("%.2f", maxDistPx) + " px)")

// 8) Simply call QuPath's helper:
setPixelSizeMicrons(pixelSize, pixelSize)

// 9) Save so the new calibration is retained
//getCurrentProjectEntry().saveImageData(imageData)

print("✅ Pixel calibration updated successfully.")