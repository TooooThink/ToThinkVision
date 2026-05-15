// ToThinkVision → After Effects Project Script
// Run this script inside After Effects: File > Scripts > Run Script File
// Generated from: test_video.mp4

// Create new composition
var comp = app.project.items.addComp("ToThinkVision_test_video", 1920, 1080, 1, 3.0, 30.0);
comp.openInViewer();

// Layer: Player (obj_0000)
var solid1 = comp.layers.addSolid([1, 0.6, 0.2], "Player_obj_0000", 80, 120, 1, 3.0);
solid1.moveToBeginning();
solid1.name = "Player_obj_0000";

// Position keyframes for obj_0000
var posProp = solid1.property('ADBE Transform Group').property('ADBE Position');
posProp.setValuesAtTimes([
    0.0000,
    0.0333,
    0.0667,
    0.1000,
    0.1333,
    0.1667,
    0.2000,
    0.2333,
    0.2667,
    0.3000,
], [
    [140.00, 260.00],
    [145.00, 262.00],
    [150.00, 264.00],
    [155.00, 266.00],
    [160.00, 268.00],
    [165.00, 270.00],
    [170.00, 272.00],
    [175.00, 274.00],
    [180.00, 276.00],
    [185.00, 278.00],
]);

// Opacity for obj_0000
var opProp = solid1.property('ADBE Transform Group').property('ADBE Opacity');
opProp.setValueAtTime(0, 0);
opProp.setValueAtTime(0.0000, 100);
opProp.setValueAtTime(2.9667, 100);
opProp.setValueAtTime(2.9997, 0);

// Camera null object (from MASt3R reconstruction)
var camNull = comp.layers.addNull();
camNull.name = "Camera_Tracking";
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.0000, [0.00, 0.00, -2.00]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.0333, [0.00, 0.00, -2.50]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.0667, [0.00, 0.00, -3.00]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.1000, [0.00, 0.00, -3.50]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.1333, [0.00, 0.00, -4.00]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.1667, [0.00, 0.00, -4.50]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.2000, [0.00, 0.00, -5.00]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.2333, [0.00, 0.00, -5.50]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.2667, [0.00, 0.00, -6.00]);
camNull.property("ADBE Transform Group").property("ADBE Position").setValueAtTime(0.3000, [0.00, 0.00, -6.50]);

// Import point cloud as shape layer (if available)
// Point cloud: 100 points

// Done!
alert("ToThinkVision project imported successfully!");