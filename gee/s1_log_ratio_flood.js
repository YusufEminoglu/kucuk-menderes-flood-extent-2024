/*******************************************************
 * Sentinel-1 log-ratio flood mapping
 * ROI = HydroSHEDS Level-7 basin containing a user point
 *******************************************************/

// ==== 0) USER INPUTS ====
if (typeof geometry === 'undefined') {
  throw new Error('Draw ONE point named `geometry` in the editor (Imports pane).');
}

var beforeStart = '2024-09-01';
var beforeEnd   = '2024-09-09';
var afterStart  = '2024-09-10';
var afterEnd    = '2024-09-19';
var polarization = 'VH';
var SLOPE_THRESHOLD = 5;
var CONNECTED_PIXELS = 8;
var EXPORT_CRS = 'EPSG:5253';
var EXPORT_FOLDER = 'FloodMapping';

// ==== 1) ROI: LEVEL-7 BASIN FROM POINT ====
var point = ee.Geometry(geometry);
var basinsL7 = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7');

// Basin that contains the point
var basinL7 = basinsL7.filterBounds(point).first();

// Optional guard (client-side) in case point is outside coverage
basinL7.evaluate(function(f){
  if (f === null) {
    throw new Error('No Level-7 basin found for the provided point.');
  }
});

var basinId  = basinL7.get('HYBAS_ID');
var pfafId   = basinL7.get('PFAF_ID');
var roi      = basinL7.geometry();

print('Selected Level-7 basin → HYBAS_ID:', basinId, ', PFAF_ID:', pfafId);

// Quick map check
Map.centerObject(roi, 9);
Map.addLayer(roi, {color: 'yellow', strokeWidth: 2, fillOpacity: 0}, 'ROI: L7 basin');

// ==== 2) HELPERS ====
function toNatural(img) { return ee.Image(10).pow(img.select(0).divide(10)); }
function toDB(img)      { return ee.Image(img).log10().multiply(10); }

// Refined Lee speckle filter (unchanged; single-band)
function RefinedLee(img) {
  var weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3);
  var kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, false);
  var mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
  var variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);

  var sample_weights = ee.List([
    [0,0,0,0,0,0,0],
    [0,1,0,1,0,1,0],
    [0,0,0,0,0,0,0],
    [0,1,0,1,0,1,0],
    [0,0,0,0,0,0,0],
    [0,1,0,1,0,1,0],
    [0,0,0,0,0,0,0]
  ]);
  var sample_kernel = ee.Kernel.fixed(7,7,sample_weights,3,3,false);
  var sample_mean = mean3.neighborhoodToBands(sample_kernel);
  var sample_var  = variance3.neighborhoodToBands(sample_kernel);

  var gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
  gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
  gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
  gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
  var max_gradient = gradients.reduce(ee.Reducer.max());
  var gradmask = gradients.eq(max_gradient);
  gradmask = gradmask.addBands(gradmask);

  var directions = sample_mean.select(1).subtract(sample_mean.select(4))
    .gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
  directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4))
    .gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
  directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4))
    .gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
  directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4))
    .gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
  directions = directions.addBands(directions.select(0).not().multiply(5));
  directions = directions.addBands(directions.select(1).not().multiply(6));
  directions = directions.addBands(directions.select(2).not().multiply(7));
  directions = directions.addBands(directions.select(3).not().multiply(8));
  directions = directions.updateMask(gradmask);
  directions = directions.reduce(ee.Reducer.sum());

  var sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
  var sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5)
    .arrayReduce(ee.Reducer.mean(),[0]);

  var rect_weights = ee.List.repeat(ee.List.repeat(0,7),3)
    .cat(ee.List.repeat(ee.List.repeat(1,7),4));
  var diag_weights = ee.List([
    [1,0,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0],
    [1,1,1,1,1,0,0],
    [1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1]
  ]);
  var rect_kernel = ee.Kernel.fixed(7,7,rect_weights,3,3,false);
  var diag_kernel = ee.Kernel.fixed(7,7,diag_weights,3,3,false);

  var dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
  var dir_var  = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
  dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
  dir_var  = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
  for (var i=1; i<4; i++) {
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_var  = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
    dir_var  = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
  }
  dir_mean = dir_mean.reduce(ee.Reducer.sum());
  dir_var  = dir_var.reduce(ee.Reducer.sum());

  var varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1));
  var b    = varX.divide(dir_var);
  var result = dir_mean.add(b.multiply(img.subtract(dir_mean)));
  return result.arrayFlatten([['sum']]);
}

// Otsu helper
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means  = ee.Array(histogram.get('bucketMeans'));
  var size   = means.length().get([0]);
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum    = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean   = sum.divide(total);
  var idx = ee.List.sequence(1, size);
  var bss = idx.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount  = total.subtract(aCount);
    var bMean   = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
      .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.sort(bss).get([-1]);
}

// ==== 3) DATASETS ====
var gsw = ee.Image('JRC/GSW1_3/GlobalSurfaceWater');
var hydroshedsDEM = ee.Image('WWF/HydroSHEDS/03VFDEM');

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filterBounds(roi);

var beforeCollection = s1.filter(ee.Filter.date(beforeStart, beforeEnd));
var afterCollection  = s1.filter(ee.Filter.date(afterStart,  afterEnd));

var preEventSar  = beforeCollection.mosaic().clip(roi);
var postEventSar = afterCollection.mosaic().clip(roi);

// Natural units + speckle filtering
var preNatural  = RefinedLee(toNatural(preEventSar));
var postNatural = RefinedLee(toNatural(postEventSar));

// Log-ratio (dB)
var logRatio = toDB(postNatural.divide(preNatural)).rename('log_ratio');

// Histogram for Otsu
var histogram = logRatio.reduceRegion({
  reducer: ee.Reducer.histogram(255),
  geometry: roi,
  scale: 30,
  maxPixels: 1e13
}).get('log_ratio');

var threshold = ee.Number(ee.Algorithms.If(histogram, otsu(histogram), 999));

// Initial flood
var initialFlood = logRatio.gt(threshold).selfMask()
  .updateMask(threshold.neq(999));

// Masks
var permanentWaterMask = gsw.select('seasonality').gte(5).not();
var slopeMask = ee.Algorithms.Terrain(hydroshedsDEM).select('slope').lt(SLOPE_THRESHOLD);
var connections = initialFlood.connectedPixelCount(100, false);
var connectionsMask = connections.gte(CONNECTED_PIXELS);

// Final flood extent
var finalFlood = initialFlood
  .updateMask(permanentWaterMask)
  .updateMask(slopeMask)
  .updateMask(connectionsMask);

// ==== 4) DISPLAY ====
threshold.evaluate(function(val, error) {
  if (error) {
    print('Error computing threshold:', error);
  } else if (val === 999) {
    print('Threshold not calculated (histogram empty?).');
  } else {
    print('Otsu threshold:', val);
    var sarVis  = {min: -25, max: 0};
    var diffVis = {min: -5, max: 10, palette: ['#000000','#ffffff','#377eb8']};
    Map.addLayer(preEventSar,  sarVis, 'Pre-event SAR', false);
    Map.addLayer(postEventSar, sarVis, 'Post-event SAR', false);
    Map.addLayer(logRatio,     diffVis, 'Log-ratio', false);
    Map.addLayer(initialFlood, {palette: ['#ff7f0e']}, 'Binary flood mask', false);
    Map.addLayer(finalFlood,   {palette: ['#1f78b4']}, 'Final flood extent', true);
  }
});

// Histogram chart
var histogramChart = ui.Chart.image.histogram({
  image: logRatio,
  region: roi,
  scale: 100
}).setOptions({
  title: 'SAR log-ratio histogram',
  hAxis: {title: 'Log-ratio (dB)'},
  vAxis: {title: 'Pixel count'},
  series: {0: {color: 'blue'}}
});
print(histogramChart);

// ==== 5) EXPORTS ====
Export.image.toDrive({
  image: preEventSar.toFloat(),
  description: 'PreEventSAR',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: postEventSar.toFloat(),
  description: 'PostEventSAR',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: logRatio.toFloat(),
  description: 'LogRatio',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: initialFlood.toByte(),
  description: 'BinaryFloodMask',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: finalFlood.toByte(),
  description: 'FinalFloodExtent',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: slopeMask.toByte(),
  description: 'SlopeMask',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: permanentWaterMask.toByte(),
  description: 'PermanentWaterMask',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: connectionsMask.toByte(),
  description: 'ConnectionsMask',
  folder: EXPORT_FOLDER,
  crs: EXPORT_CRS,
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
