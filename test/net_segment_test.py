import net_segment.obj as seg
import imagej
import code

# initialize imagej
print("Initializing ImageJ...")
ij = imagej.init(headless=False)
print(f"ImageJ version: {ij.getVersion()}")

x = seg.open_image('/home/edward/Documents/repos/personal/net_segment/doc/sample-data/test_8bit.tif', show=True)

# return interperter
code.interact(local=locals())