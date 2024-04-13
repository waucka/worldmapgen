# worldmapgen

This is something I threw together a while back for the purpose of experimentation.  I make zero guarantees about its quality or usefulness.

The viewer program is currently broken because wgpu no longer accepts the vertex shader I wrote.  I was unaware at the time that sampling a
texture from the vertex shader was frowned upon.

## Running

You can do `worldgencli terraingen --output heightmap.png` to get a heightmap.
