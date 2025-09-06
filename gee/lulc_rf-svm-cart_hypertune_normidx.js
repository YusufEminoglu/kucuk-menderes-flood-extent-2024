// ==== REQUIRED INPUTS ====
// ROI: draw/import as `roi` (preferred). Fallbacks: `r1`, `geometry`.
var _roi = (typeof roi !== 'undefined') ? roi
  : (typeof r1 !== 'undefined') ? ee.FeatureCollection(r1).geometry()
  : (typeof geometry !== 'undefined') ? ee.Geometry(geometry)
  : null;
if (_roi === null) { throw new Error('Provide ROI as `roi` polygon (or `r1` / `geometry`).'); }
var ROI = ee.Geometry(_roi);

// GCP FeatureCollection: provide as `gcp` with a `landcover` column. Fallback: `table`.
var _gcp = (typeof gcp !== 'undefined') ? gcp
  : (typeof table !== 'undefined') ? table
  : null;
if (_gcp === null) { throw new Error('Provide GCP FeatureCollection as `gcp` (or `table`) with a `landcover` field.'); }
var GCP = ee.FeatureCollection(_gcp);

// ==== PARAMETERS ====
var OrderedScale = 10;
var s2DateStart = '2022-01-01';
var s2DateEnd   = '2023-01-01';
var cloudPctMax = 30;

// ==== HELPERS ====
function maskCloudAndShadowsSR(image){
  var scl = image.select('SCL');
  var cloud = image.select('MSK_CLDPRB').lt(10);
  var cirrus = scl.eq(10);
  var shadow = scl.eq(3);
  var mask = cloud.and(cirrus.neq(1)).and(shadow.neq(1));
  return image.updateMask(mask);
}
function addIndices(image){
  var ndvi  = image.normalizedDifference(['B8','B4']).rename('ndvi');
  var ndbi  = image.normalizedDifference(['B11','B8']).rename('ndbi');
  var mndwi = image.normalizedDifference(['B3','B11']).rename('mndwi');
  var bsi   = image.expression('((X+Y)-(A+B))/((X+Y)+(A+B))',{
    X:image.select('B11'),Y:image.select('B4'),A:image.select('B8'),B:image.select('B2')
  }).rename('bsi');
  return image.addBands([ndvi,ndbi,mndwi,bsi]);
}
function normalize(image){
  var minMax = image.reduceRegion({
    reducer: ee.Reducer.min().combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
    geometry: ROI, scale: OrderedScale, maxPixels: 1e10, bestEffort: true, tileScale: 16
  });
  var bands = image.bandNames();
  var mins = ee.Image.constant(bands.map(function(b){ return minMax.get(ee.String(b).cat('_min')); }));
  var maxs = ee.Image.constant(bands.map(function(b){ return minMax.get(ee.String(b).cat('_max')); }));
  return image.subtract(mins).divide(maxs.subtract(mins));
}
function exportTable(fc, desc, prefix){
  Export.table.toDrive({collection: fc, description: desc, folder: 'earthengine', fileNamePrefix: prefix, fileFormat: 'CSV'});
}
function exportImage(image, desc){
  Export.image.toDrive({image: image.clip(ROI), description: desc, scale: 10, region: ROI, crs: 'EPSG:5253', fileFormat: 'GeoTIFF', maxPixels: 1e10});
}

// ==== DATASETS ====
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPctMax))
  .filterDate(s2DateStart, s2DateEnd)
  .filterBounds(ROI)
  .map(maskCloudAndShadowsSR)
  .select('B.*');

var composite = addIndices(s2.median());
var alos = ee.Image('JAXA/ALOS/AW3D30/V2_2');
var elev = alos.select('AVE_DSM').rename('elev');
var slope = ee.Terrain.slope(alos.select('AVE_DSM')).rename('slope');
composite = composite.addBands([elev, slope]);
composite = normalize(composite);

// ==== SPLIT ====
var GCPspl = GCP.randomColumn('random', 42);
var trainingGcp   = GCPspl.filter(ee.Filter.lt('random', 0.7));
var validationGcp = GCPspl.filter(ee.Filter.gte('random', 0.7));

var training = composite.sampleRegions({
  collection: trainingGcp, properties: ['landcover'], scale: OrderedScale, tileScale: 16
});
var test = composite.sampleRegions({
  collection: validationGcp, properties: ['landcover'], scale: OrderedScale, tileScale: 16
});

// ==== HYPERPARAM TUNING: RF ====
var numTreesList = ee.List.sequence(10, 150, 10);
var varsPerSplitList = ee.List.sequence(1, composite.bandNames().size());
var rfGrid = numTreesList.map(function(nt){
  return varsPerSplitList.map(function(vps){
    var cls = ee.Classifier.smileRandomForest({numberOfTrees: nt, variablesPerSplit: vps})
      .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});
    var acc = test.classify(cls).errorMatrix('landcover','classification').accuracy();
    return ee.Feature(null, {accuracy: acc, numberOfTrees: nt, variablesPerSplit: vps});
  });
}).flatten();
var rfFc = ee.FeatureCollection(rfGrid);
exportTable(rfFc, 'RandomForest_Hyperparameter_Tuning', 'rf_numtrees_varspl');
var rfBest = rfFc.sort('accuracy', false).first();
var optimalNumTrees = rfBest.getNumber('numberOfTrees');
var optimalVarsPerSplit = rfBest.getNumber('variablesPerSplit');

// ==== HYPERPARAM TUNING: SVM ====
var kernelTypeList = ee.List(['LINEAR','POLY','RBF','SIGMOID']);
var costList = ee.List.sequence(0.1, 10, 0.1);
var svmGrid = kernelTypeList.map(function(k){
  return costList.map(function(c){
    var cls = ee.Classifier.libsvm({kernelType: k, cost: c})
      .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});
    var acc = test.classify(cls).errorMatrix('landcover','classification').accuracy();
    return ee.Feature(null, {accuracy: acc, kernelType: k, cost: c});
  });
}).flatten();
var svmFc = ee.FeatureCollection(svmGrid);
exportTable(svmFc, 'SVM_Hyperparameter_Tuning', 'svm_kernel_cost');
var svmBest = svmFc.sort('accuracy', false).first();
var optimalKernelType = svmBest.getString('kernelType');
var optimalCost = svmBest.getNumber('cost');

// ==== HYPERPARAM TUNING: CART ====
var maxNodesList = ee.List.sequence(10, 100, 10);
var cartGrid = maxNodesList.map(function(mn){
  var cls = ee.Classifier.smileCart({maxNodes: mn})
    .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});
  var acc = test.classify(cls).errorMatrix('landcover','classification').accuracy();
  return ee.Feature(null, {accuracy: acc, maxNodes: mn});
});
var cartFc = ee.FeatureCollection(cartGrid);
exportTable(cartFc, 'CART_Hyperparameter_Tuning', 'cart_maxnodes');
var cartBest = cartFc.sort('accuracy', false).first();
var optimalMaxNodes = cartBest.getNumber('maxNodes');

// ==== FINAL MODELS ====
var RF = ee.Classifier.smileRandomForest({numberOfTrees: optimalNumTrees, variablesPerSplit: optimalVarsPerSplit})
  .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});
var SVM = ee.Classifier.libsvm({kernelType: optimalKernelType, cost: optimalCost, decisionProcedure: 'Margin'})
  .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});
var CART = ee.Classifier.smileCart({maxNodes: optimalMaxNodes})
  .train({features: training, classProperty: 'landcover', inputProperties: composite.bandNames()});

var clsRF   = composite.classify(RF);
var clsSVM  = composite.classify(SVM);
var clsCART = composite.classify(CART);

Map.centerObject(ROI);
Map.addLayer(composite.select(['B4','B3','B2']).clip(ROI), {min:0,max:3000,gamma:1.2}, 'RGB', false);
Map.addLayer(clsRF.clip(ROI),   {min:0,max:4,palette:['#d63000','#f5ffd9','#42b5ff','#a1ff52','#427246']}, 'RF (final)', false);
Map.addLayer(clsSVM.clip(ROI),  {min:0,max:4,palette:['#d63000','#f5ffd9','#42b5ff','#a1ff52','#427246']}, 'SVM (final)', false);
Map.addLayer(clsCART.clip(ROI), {min:0,max:4,palette:['#d63000','#f5ffd9','#42b5ff','#a1ff52','#427246']}, 'CART (final)', false);

// ==== ACCURACY + MATRICES ====
function assess(classifiedImg, name){
  var val = classifiedImg.sampleRegions({collection: validationGcp, properties:['landcover'], scale: OrderedScale, tileScale:16});
  var cm = val.errorMatrix('landcover','classification');
  var res = ee.Feature(null, {
    classifier: name,
    overallAccuracy: cm.accuracy(),
    kappa: cm.kappa(),
    precision: cm.producersAccuracy(),
    recall: cm.consumersAccuracy(),
    f1Score: cm.fscore()
  });
  Export.table.toDrive({
    collection: ee.FeatureCollection([ee.Feature(null,{classifier:name, matrix: cm.array()})]),
    description: name + '_Error_Matrix',
    folder: 'earthengine',
    fileNamePrefix: name.toLowerCase() + '_error_matrix',
    fileFormat: 'CSV'
  });
  return res;
}
var rfRes   = assess(clsRF, 'RF');
var svmRes  = assess(clsSVM, 'SVM');
var cartRes = assess(clsCART, 'CART');
exportTable(ee.FeatureCollection([rfRes]),   'RF_Accuracy_Assessment',   'rf_accuracy_assessment');
exportTable(ee.FeatureCollection([svmRes]),  'SVM_Accuracy_Assessment',  'svm_accuracy_assessment');
exportTable(ee.FeatureCollection([cartRes]), 'CART_Accuracy_Assessment', 'cart_accuracy_assessment');

// ==== FEATURE IMPORTANCE (where available) ====
function exportFeatureImportance(classifier, modelName){
  var exp = classifier.explain();
  var hasImp = exp.contains('importance');
  var imp = ee.Dictionary(exp.get('importance'));
  var sum = imp.values().reduce(ee.Reducer.sum());
  var rel = imp.map(function(k,v){ return ee.Number(v).multiply(100).divide(sum); });
  var fc = ee.FeatureCollection(rel.keys().map(function(k){ return ee.Feature(null, {band:k, importance: rel.get(k)}); }));
  Export.table.toDrive({collection: fc, description: modelName + '_Feature_Importance', folder:'earthengine', fileNamePrefix: modelName.toLowerCase() + '_feature_importance', fileFormat:'CSV'});
}
exportFeatureImportance(RF, 'RF');
exportFeatureImportance(CART, 'CART'); // SVM typically lacks per-feature importance

// ==== SCORE EXPORTS ====
function exportScores(classifier, modelName, outputMode){
  var scoreImg = composite.classify(classifier.setOutputMode(outputMode));
  var scores = scoreImg.sampleRegions({collection: validationGcp, properties:['landcover'], scale: OrderedScale, tileScale:16});
  exportTable(scores, modelName + '_Test_Scores', modelName.toLowerCase() + '_test_scores');
}
exportScores(RF,   'RF',   'MULTIPROBABILITY');
exportScores(CART, 'CART', 'MULTIPROBABILITY');
exportScores(SVM,  'SVM',  'RAW');

// ==== SAMPLED NORMALIZED PREDICTORS (keep split tag) ====
var sampled = composite.sampleRegions({collection: GCPspl, properties:['landcover','random'], scale: OrderedScale, tileScale: 16});
exportTable(sampled, 'Sampled_Normalized_Parameters', 'sampled_normalized_parameters');

// ==== EXPORT RASTERS ====
exportImage(clsRF.toFloat(),   'FinalRFClassification');
exportImage(clsSVM.toFloat(),  'FinalSVMClassification');
exportImage(clsCART.toFloat(), 'FinalCARTClassification');

