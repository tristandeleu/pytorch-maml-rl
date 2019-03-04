Python bindings for Optimal Reciprocal Collision Avoidance
==========================================================

This repository contains the RVO2 framework, as described below, along with
[Cython](http://cython.org/)-based Python bindings. Its home is
[GitHub](https://github.com/sybrenstuvel/Python-RVO2). New updates are released
there. There are no explicit version numbers -- all commits on the master
branch are supposed to be stable.


Building & installing
----------------------

Building requires [CMake](http://cmake.org/) and [Cython](http://cython.org/) to be installed.
Run `pip install -r requirements.txt` to install the tested version of Cython, or run
`pip install Cython` to install the latest version.

Run `python setup.py build` to build, and `python setup.py install` to install.
Alternatively, if you want an in-place build that puts the compiled library right in
the current directory, run `python setup.py build_ext --inplace`

Only tested with Python 2.7, 3.4, and 3.6 on Ubuntu Linux. The setup.py script uses CMake to build
the RVO2 library itself, before building the Python wrappers. If you have success (or failure)
stories, please share them!

To build on Mac OSX, give an `export MACOSX_DEPLOYMENT_TARGET=10.xx` command first, before
running `python setup.py build`. Replace `10.xx` with your version of OSX, for example `10.11`.

Differences with the C++ version
--------------------------------

The `Vector2` and `Line` classes from the RVO2 library are _not_ wrapped. Instead,
vectors are passed as tuples `(x, y)` from/to Python. Lines are passed as tuples
`(point x, point y, direction x, direction y)`.


Example code
------------

```python
#!/usr/bin/env python

import rvo2

sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

for step in range(20):
    sim.doStep()

    positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
                 for agent_no in (a0, a1, a2, a3)]
    print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
```


Threading support
--------------------------------

Calling Python-RVO2 from multiple threads has not been tested. However, code that
may take longer to run (`doStep()`, `processObstacles()` and `queryVisibility(...)`)
release the Global Interpreter Lock (GIL) so that other Python threads can run while
RVO2 is processing.


Optimal Reciprocal Collision Avoidance
======================================

<http://gamma.cs.unc.edu/RVO2/>

[![Build Status](https://travis-ci.org/snape/RVO2.svg?branch=master)](https://travis-ci.org/snape/RVO2)
[![Build status](https://ci.appveyor.com/api/projects/status/0nyp7y4di8x1gh9o/branch/master?svg=true)](https://ci.appveyor.com/project/snape/rvo2)

Copyright 2008 University of North Carolina at Chapel Hill

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Please send all bug reports for the Python wrapper to
[Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2), and bug
report for the RVO2 library itself to [geom@cs.unc.edu](mailto:geom@cs.unc.edu).

The RVO2 authors may be contacted via:

Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, and Dinesh Manocha  
Dept. of Computer Science  
201 S. Columbia St.  
Frederick P. Brooks, Jr. Computer Science Bldg.  
Chapel Hill, N.C. 27599-3175  
United States of America
