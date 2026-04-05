import sys
try:
    import rasterio
    with rasterio.open(sys.argv[1]) as src:
        print("Driver:", src.driver)
        print("Width:", src.width, "Height:", src.height)
        print("Count:", src.count)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bounds:", src.bounds)
except ImportError:
    try:
        from osgeo import gdal
        ds = gdal.Open(sys.argv[1])
        print("Driver:", ds.GetDriver().ShortName)
        print("Width:", ds.RasterXSize, "Height:", ds.RasterYSize)
        print("Count:", ds.RasterCount)
        print("Projection:", ds.GetProjection())
        print("Transform:", ds.GetGeoTransform())
    except ImportError:
        print("Neither rasterio nor gdal is installed.")
