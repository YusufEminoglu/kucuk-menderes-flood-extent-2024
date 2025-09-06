// ROI: draw/import a polygon or feature named `roi` (preferred). Fallbacks: `r1`, `geometry`.
var _roi = (typeof roi !== 'undefined') ? roi
  : (typeof r1 !== 'undefined') ? ee.FeatureCollection(r1).geometry()
  : (typeof geometry !== 'undefined') ? ee.Geometry(geometry)
  : null;
if (_roi === null) { throw new Error('Provide an ROI: draw/import `roi` (polygon), or define `r1` / `geometry`.'); }
var ROI = ee.Geometry(_roi);

// ---- Core helpers ----
function toNatural(img){ return ee.Image(10).pow(img.select(0).divide(10)); }
function toDB(img){ return ee.Image(img).log10().multiply(10); }
function RefinedLee(img){
  var w3 = ee.List.repeat(ee.List.repeat(1,3),3);
  var k3 = ee.Kernel.fixed(3,3,w3,1,1,false);
  var m3 = img.reduceNeighborhood(ee.Reducer.mean(),k3);
  var v3 = img.reduceNeighborhood(ee.Reducer.variance(),k3);
  var sw = ee.List([[0,0,0,0,0,0,0],[0,1,0,1,0,1,0],[0,0,0,0,0,0,0],[0,1,0,1,0,1,0],[0,0,0,0,0,0,0],[0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
  var sk = ee.Kernel.fixed(7,7,sw,3,3,false);
  var sm = m3.neighborhoodToBands(sk);
  var sv = v3.neighborhoodToBands(sk);
  var g = sm.select(1).subtract(sm.select(7)).abs()
           .addBands(sm.select(6).subtract(sm.select(2)).abs())
           .addBands(sm.select(3).subtract(sm.select(5)).abs())
           .addBands(sm.select(0).subtract(sm.select(8)).abs());
  var gmax = g.reduce(ee.Reducer.max());
  var gmask = g.eq(gmax).addBands(g.eq(gmax));
  var d = sm.select(1).subtract(sm.select(4)).gt(sm.select(4).subtract(sm.select(7))).multiply(1);
  d = d.addBands(sm.select(6).subtract(sm.select(4)).gt(sm.select(4).subtract(sm.select(2))).multiply(2));
  d = d.addBands(sm.select(3).subtract(sm.select(4)).gt(sm.select(4).subtract(sm.select(5))).multiply(3));
  d = d.addBands(sm.select(0).subtract(sm.select(4)).gt(sm.select(4).subtract(sm.select(8))).multiply(4));
  d = d.addBands(d.select(0).not().multiply(5))
       .addBands(d.select(1).not().multiply(6))
       .addBands(d.select(2).not().multiply(7))
       .addBands(d.select(3).not().multiply(8))
       .updateMask(gmask)
       .reduce(ee.Reducer.sum());
  var sstats = sv.divide(sm.multiply(sm));
  var sigmaV = sstats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(),[0]);
  var rw = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));
  var dw = ee.List([[1,0,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,1,0,0,0,0],[1,1,1,1,0,0,0],[1,1,1,1,1,0,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,1]]);
  var rk = ee.Kernel.fixed(7,7,rw,3,3,false);
  var dk = ee.Kernel.fixed(7,7,dw,3,3,false);
  var dm = img.reduceNeighborhood(ee.Reducer.mean(),rk).updateMask(d.eq(1));
  var dv = img.reduceNeighborhood(ee.Reducer.variance(),rk).updateMask(d.eq(1));
  dm = dm.addBands(img.reduceNeighborhood(ee.Reducer.mean(),dk).updateMask(d.eq(2)));
  dv = dv.addBands(img.reduceNeighborhood(ee.Reducer.variance(),dk).updateMask(d.eq(2)));
  for (var i=1;i<4;i++){
    dm = dm.addBands(img.reduceNeighborhood(ee.Reducer.mean(),rk.rotate(i)).updateMask(d.eq(2*i+1)));
    dv = dv.addBands(img.reduceNeighborhood(ee.Reducer.variance(),rk.rotate(i)).updateMask(d.eq(2*i+1)));
    dm = dm.addBands(img.reduceNeighborhood(ee.Reducer.mean(),dk.rotate(i)).updateMask(d.eq(2*i+2)));
    dv = dv.addBands(img.reduceNeighborhood(ee.Reducer.variance(),dk.rotate(i)).updateMask(d.eq(2*i+2)));
  }
  dm = dm.reduce(ee.Reducer.sum());
  dv = dv.reduce(ee.Reducer.sum());
  var varX = dv.subtract(dm.multiply(dm).multiply(sigmaV)).divide(sigmaV.add(1));
  var b = varX.divide(dv);
  var out = dm.add(b.multiply(img.subtract(dm)));
  return out.arrayFlatten([['sum']]);
}

// ---- Datasets & parameters ----
var beforeStart = '2024-08-20', beforeEnd = '2024-09-08';
var afterStart  = '2024-09-10', afterEnd  = '2024-09-24';
var polarization = 'VH', passDirection = 'DESCENDING';
var diffThreshold = 1.25, slopeThreshold = 5, connectedPixelThreshold = 8;

var DEM = ee.Image('WWF/HydroSHEDS/03VFDEM');
var GSW = ee.Image('JRC/GSW1_3/GlobalSurfaceWater');
var HRSL = ee.ImageCollection('projects/sat-io/open-datasets/hrsl/hrslpop').median().unmask(0);
var ESAWC = ee.ImageCollection('ESA/WorldCover/v200').median().clip(ROI);

// ---- SAR stacks ----
var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation',polarization))
  .filter(ee.Filter.eq('orbitProperties_pass',passDirection))
  .filter(ee.Filter.eq('resolution_meters',10))
  .filterBounds(ROI)
  .select(polarization);

var before = S1.filter(ee.Filter.date(beforeStart,beforeEnd)).mosaic().clip(ROI);
var after  = S1.filter(ee.Filter.date(afterStart,afterEnd)).mosaic().clip(ROI);

// ---- Filtering in natural units; ratio in linear space ----
var beforeNat = RefinedLee(toNatural(before));
var afterNat  = RefinedLee(toNatural(after));
var ratioLin  = afterNat.divide(beforeNat);

// ---- Thresholding & masks ----
var flooded0 = ratioLin.gt(diffThreshold).rename('water').selfMask();
var waterMask = GSW.select('seasonality').gte(5).clip(ROI).unmask(0).not();
var slope = ee.Algorithms.Terrain(DEM).select('slope');
var slopeMask = slope.lte(slopeThreshold);
var conn = flooded0.connectedPixelCount(100,false).gte(connectedPixelThreshold);
var flooded = flooded0.updateMask(waterMask).updateMask(slopeMask).updateMask(conn);

// ---- Stats ----
Map.centerObject(ROI,10);
Map.addLayer(ROI,{color:'gray',strokeWidth:2,fillOpacity:0},'ROI',false);
Map.addLayer(before,{min:-25,max:0},'Before (dB)',false);
Map.addLayer(after,{min:-25,max:0},'After (dB)',false);
Map.addLayer(toDB(beforeNat),{min:-25,max:0},'Before (filtered dB)',false);
Map.addLayer(toDB(afterNat),{min:-25,max:0},'After (filtered dB)',false);
Map.addLayer(flooded,{palette:['#1f78b4']},'Flooded');

var roiAreaHa = ROI.area().divide(1e4).round();
print('ROI area (ha):', roiAreaHa);

var floodAreaM2 = flooded.multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: ROI, scale: 10, maxPixels: 1e10, tileScale: 16
});
var floodAreaHa = ee.Number(floodAreaM2.get('water')).divide(1e4).round();
print('Flooded area (ha):', floodAreaHa);

// ---- Exposure: population ----
var pop = HRSL.clip(ROI);
var popExposed = pop.updateMask(flooded).updateMask(pop);
var popSum = popExposed.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: ROI, scale: 30, maxPixels: 1e10, tileScale: 16
});
print('Exposed population:', ee.Number(popSum.get('b1')).round());

// ---- Exposure: land cover classes ----
var lc = ESAWC.select('Map');
var lcCodes = {'Tree cover':10,'Shrubland':20,'Grassland':30,'Cropland':40,'Built-up':50,'Bare / sparse vegetation':60};

function affectedClass(code,val){
  var mask = lc.eq(code);
  var affected = flooded.updateMask(mask);
  var area = affected.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: ROI, scale: 10, maxPixels: 1e10, tileScale: 16
  });
  print('Affected ' + code + ' (ha):', ee.Number(area.get('water')).divide(1e4).round());
  return affected.multiply(ee.Image.constant(val));
}

var affectedComposite = ee.Image(0);
var keys = Object.keys(lcCodes);
for (var i=0;i<keys.length;i++){
  var codeName = keys[i], codeVal = lcCodes[codeName], classVal = i+1;
  affectedComposite = affectedComposite.add(affectedClass(codeVal,classVal)).unmask(affectedComposite);
}

Map.addLayer(affectedComposite.clip(ROI),
             {min:0,max:6,palette:['#000000','#1a9850','#fee08b','#fdae61','#d7191c','#7b3294','#2c7fb8']},
             'Combined affected areas', false);

// ---- Exports ----
Export.image.toDrive({image: toDB(beforeNat).toFloat(), description:'Before_filtered_dB', scale:10, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: toDB(afterNat).toFloat(),  description:'After_filtered_dB',  scale:10, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: ratioLin.toFloat(),        description:'SAR_linear_ratio',   scale:10, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: flooded.toByte(),          description:'Flooded_mask',       scale:10, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: pop.toFloat(),             description:'Population',         scale:30, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: popExposed.toFloat(),      description:'Population_exposed', scale:30, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: lc.toByte(),               description:'ESA_WorldCover',     scale:10, region: ROI, maxPixels:1e10});
Export.image.toDrive({image: affectedComposite.toByte(),description:'Combined_affected_areas', scale:10, region: ROI, maxPixels:1e10, fileFormat:'GEOTIFF'});
