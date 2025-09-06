var _roi = (typeof roi !== 'undefined') ? roi
  : (typeof r1 !== 'undefined') ? ee.FeatureCollection(r1).geometry()
  : (typeof geometry !== 'undefined') ? ee.Geometry(geometry)
  : null;
if (_roi === null) { throw new Error('Provide ROI as `roi` polygon (or `r1` / `geometry`).'); }
var ROI = roi;

var DATE_START = '2024-09-01';
var DATE_END   = '2024-09-30';
var CLOUDY_PCT = 30;
var SCALE = 10;
var EDGE_SIGMA = 1.2;
var EDGE_THRESH = 0.08;
var EDGE_BUFFER_PX = 6;
var MIN_CONN = 8;

function maskS2sr(img){
  var scl = img.select('SCL');
  var mask = img.select('MSK_CLDPRB').lt(10)
    .and(scl.neq(3)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask);
}
function otsu(dict){
  dict = ee.Dictionary(dict);
  var h = ee.Array(dict.get('histogram'));
  var m = ee.Array(dict.get('bucketMeans'));
  var n = m.length().get([0]);
  var total = h.reduce(ee.Reducer.sum(),[0]).get([0]);
  var sum = m.multiply(h).reduce(ee.Reducer.sum(),[0]).get([0]);
  var mean = sum.divide(total);
  var idx = ee.List.sequence(1,n);
  var bss = idx.map(function(i){
    var aH = h.slice(0,0,i), aC = aH.reduce(ee.Reducer.sum(),[0]).get([0]);
    var aM = m.slice(0,0,i);
    var aMean = aM.multiply(aH).reduce(ee.Reducer.sum(),[0]).get([0]).divide(aC);
    var bC = total.subtract(aC);
    var bMean = sum.subtract(aC.multiply(aMean)).divide(bC);
    return aC.multiply(aMean.subtract(mean).pow(2)).add(bC.multiply(bMean.subtract(mean).pow(2)));
  });
  return m.sort(bss).get([-1]);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ROI)
  .filterDate(DATE_START, DATE_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PCT))
  .map(maskS2sr)
  .select(['B2','B3','B4','B8','B11','SCL']);

var img = s2.median().clip(ROI);
var mndwi = img.normalizedDifference(['B3','B11']).rename('mndwi');

var histGlobal = mndwi.reduceRegion({
  reducer: ee.Reducer.histogram(512), geometry: ROI, scale: SCALE, maxPixels: 1e13
}).get('mndwi');
var Tglobal = ee.Number(ee.Algorithms.If(histGlobal, otsu(histGlobal), 999));

var edges = ee.Algorithms.CannyEdgeDetector({image: mndwi, threshold: EDGE_THRESH, sigma: EDGE_SIGMA});
var edgeMask = edges.focal_max(1).reduceNeighborhood({
  reducer: ee.Reducer.max(),
  kernel: ee.Kernel.circle(EDGE_BUFFER_PX)
}).gt(0);

var histEdge = mndwi.updateMask(edgeMask).reduceRegion({
  reducer: ee.Reducer.histogram(512), geometry: ROI, scale: SCALE, maxPixels: 1e13, tileScale: 16
}).get('mndwi');
var Tedge = ee.Number(ee.Algorithms.If(histEdge, otsu(histEdge), Tglobal));

var waterGlobal = mndwi.gt(Tglobal).selfMask();
var waterEdge = mndwi.gt(Tedge).selfMask();
var water = waterGlobal.where(edgeMask, waterEdge);
var conn = water.connectedPixelCount(100,false).gte(MIN_CONN);
water = water.updateMask(conn);

Map.centerObject(ROI, 11);
Map.addLayer(ROI, {color:'#999999', fillOpacity:0}, 'ROI', false);
Map.addLayer(mndwi, {min:-0.6, max:0.6, palette:['#762a83','#f7f7f7','#1b7837']}, 'MNDWI', false);
Map.addLayer(edges.selfMask(), {palette:['#000000']}, 'Edges', false);
Map.addLayer(edgeMask.selfMask(), {palette:['#ffcc00']}, 'Edge buffer', false);
Map.addLayer(water, {palette:['#1f78b4']}, 'Water (Otsu+edge)');

print('Otsu_global', Tglobal);
print('Otsu_edge', Tedge);

Export.image.toDrive({image: mndwi.toFloat(), description:'MNDWI', region: ROI, scale: SCALE, crs:'EPSG:5253', maxPixels:1e13});
Export.image.toDrive({image: edges.selfMask().toByte(), description:'MNDWI_Edges', region: ROI, scale: SCALE, crs:'EPSG:5253', maxPixels:1e13});
Export.image.toDrive({image: ee.Image(edgeMask).selfMask().toByte(), description:'EdgeBuffer', region: ROI, scale: SCALE, crs:'EPSG:5253', maxPixels:1e13});
Export.image.toDrive({image: water.toByte(), description:'Water_OtsuEdge', region: ROI, scale: SCALE, crs:'EPSG:5253', maxPixels:1e13});

