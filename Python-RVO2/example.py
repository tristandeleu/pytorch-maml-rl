#!/usr/bin/env python

import rvo2

timeStep = 0.1
neighborDist = 1.0
maxNeighbors = 5
timeHorizon = 1.0
timeHorizonObst = timeHorizon
radius = 0.4
maxSpeed = 2
sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
# o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
# sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

sim.setAgentVelocity(a0, (2, 1))
print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('------Running simulation--------')

for step in range(int(timeHorizon/timeStep)):
	sim.doStep()

	positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no) for agent_no in (a0, a1, a2, a3)]
	velocities = ['(%5.3f, %5.3f)' % sim.getAgentVelocity(agent_no) for agent_no in (a0, a1, a2, a3)]
	# print(velocities)

	print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
	print(sim.getAgentNumObstacleNeighbors(a2))


