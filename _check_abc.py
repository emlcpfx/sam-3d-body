import sys
try:
    from alembic import Abc, AbcGeom
    archive = Abc.IArchive(r'D:\Work\Coding\ML_repos\headcase\input\beautiful-blonde-model-posing-in-a-bikini-suit-2025-12-17-22-42-51-utc_obj\body.abc')
    top = archive.getTop()
    for i in range(top.getNumChildren()):
        child = top.getChild(i)
        print(f'Child: {child.getName()} type: {child.getMetaData().serialize()}')
        mesh = AbcGeom.IPolyMesh(top, child.getName())
        schema = mesh.getSchema()
        print(f'  Num samples: {schema.getNumSamples()}')
        uv = schema.getUVsParam()
        print(f'  Has UVs: {uv.valid()}')
        if uv.valid():
            s = uv.getIndexedValue(0)
            print(f'  UV count: {len(s.getVals())}')
except ImportError:
    print("alembic python module not available, skipping")
    sys.exit(0)
