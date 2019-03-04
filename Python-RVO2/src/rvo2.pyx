# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "Vector2.h" namespace "RVO":
    cdef cppclass Vector2:
        Vector2() except +
        Vector2(float x, float y) except +
        float x() const
        float y() const


cdef extern from "RVOSimulator.h" namespace "RVO":
    cdef const size_t RVO_ERROR


cdef extern from "RVOSimulator.h" namespace "RVO":
    cdef cppclass Line:
        Vector2 point
        Vector2 direction


cdef extern from "RVOSimulator.h" namespace "RVO":
    cdef cppclass RVOSimulator:
        RVOSimulator()
        RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors,
                     float timeHorizon, float timeHorizonObst, float radius,
                     float maxSpeed, const Vector2 & velocity)
        size_t addAgent(const Vector2 & position)
        size_t addAgent(const Vector2 & position, float neighborDist,
                        size_t maxNeighbors, float timeHorizon,
                        float timeHorizonObst, float radius, float maxSpeed,
                        const Vector2 & velocity)
        size_t addObstacle(const vector[Vector2] & vertices)
        void doStep() nogil
        size_t getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const
        size_t getAgentMaxNeighbors(size_t agentNo) const
        float getAgentMaxSpeed(size_t agentNo) const
        float getAgentNeighborDist(size_t agentNo) const
        size_t getAgentNumAgentNeighbors(size_t agentNo) const
        size_t getAgentNumObstacleNeighbors(size_t agentNo) const
        size_t getAgentNumORCALines(size_t agentNo) const
        size_t getAgentObstacleNeighbor(size_t agentNo, size_t neighborNo) const
        const Line & getAgentORCALine(size_t agentNo, size_t lineNo) const
        const Vector2 & getAgentPosition(size_t agentNo) const
        const Vector2 & getAgentPrefVelocity(size_t agentNo) const
        float getAgentRadius(size_t agentNo) const
        float getAgentTimeHorizon(size_t agentNo) const
        float getAgentTimeHorizonObst(size_t agentNo) const
        const Vector2 & getAgentVelocity(size_t agentNo) const
        float getGlobalTime() const
        size_t getNumAgents() const
        size_t getNumObstacleVertices() const
        const Vector2 & getObstacleVertex(size_t vertexNo) const
        size_t getNextObstacleVertexNo(size_t vertexNo) const
        size_t getPrevObstacleVertexNo(size_t vertexNo) const
        float getTimeStep() const
        void processObstacles() nogil
        bool queryVisibility(const Vector2 & point1, const Vector2 & point2,
                             float radius) nogil const
        void setAgentDefaults(float neighborDist, size_t maxNeighbors,
                              float timeHorizon, float timeHorizonObst,
                              float radius, float maxSpeed,
                              const Vector2 & velocity)
        void setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors)
        void setAgentMaxSpeed(size_t agentNo, float maxSpeed)
        void setAgentNeighborDist(size_t agentNo, float neighborDist)
        void setAgentPosition(size_t agentNo, const Vector2 & position)
        void setAgentPrefVelocity(size_t agentNo, const Vector2 & prefVelocity)
        void setAgentRadius(size_t agentNo, float radius)
        void setAgentTimeHorizon(size_t agentNo, float timeHorizon)
        void setAgentTimeHorizonObst(size_t agentNo, float timeHorizonObst)
        void setAgentVelocity(size_t agentNo, const Vector2 & velocity)
        void setTimeStep(float timeStep)


cdef class PyRVOSimulator:
    cdef RVOSimulator *thisptr

    def __cinit__(self, float timeStep, float neighborDist, size_t maxNeighbors,
                  float timeHorizon, float timeHorizonObst, float radius,
                  float maxSpeed, tuple velocity=(0, 0)):
        cdef Vector2 c_velocity = Vector2(velocity[0], velocity[1])

        self.thisptr = new RVOSimulator(timeStep, neighborDist, maxNeighbors,
                                        timeHorizon, timeHorizonObst, radius,
                                        maxSpeed, c_velocity)

    def addAgent(self, tuple pos, neighborDist=None,
                 maxNeighbors=None, timeHorizon=None,
                 timeHorizonObst=None, radius=None, maxSpeed=None,
                 velocity=None):
        cdef Vector2 c_pos = Vector2(pos[0], pos[1])
        cdef Vector2 c_velocity

        if neighborDist is not None and velocity is None:
            raise ValueError("Either pass only 'pos', or pass all parameters.")

        if neighborDist is None:
            agent_nr = self.thisptr.addAgent(c_pos)
        else:
            c_velocity = Vector2(velocity[0], velocity[1])
            agent_nr = self.thisptr.addAgent(c_pos, neighborDist,
                                             maxNeighbors, timeHorizon,
                                             timeHorizonObst, radius, maxSpeed,
                                             c_velocity)

        if agent_nr == RVO_ERROR:
            raise RuntimeError('Error adding agent to RVO simulation')

        return agent_nr

    def addObstacle(self, list vertices):
        cdef vector[Vector2] c_vertices

        for x, y in vertices:
            c_vertices.push_back(Vector2(x, y))

        obstacle_nr = self.thisptr.addObstacle(c_vertices)
        if obstacle_nr == RVO_ERROR:
            raise RuntimeError('Error adding obstacle to RVO simulation')
        return obstacle_nr

    def doStep(self):
        with nogil:
            self.thisptr.doStep()

    def getAgentAgentNeighbor(self, size_t agent_no, size_t neighbor_no):
        return self.thisptr.getAgentAgentNeighbor(agent_no, neighbor_no)
    def getAgentMaxNeighbors(self, size_t agent_no):
        return self.thisptr.getAgentMaxNeighbors(agent_no)
    def getAgentMaxSpeed(self, size_t agent_no):
        return self.thisptr.getAgentMaxSpeed(agent_no)
    def getAgentNeighborDist(self, size_t agent_no):
        return self.thisptr.getAgentNeighborDist(agent_no)
    def getAgentNumAgentNeighbors(self, size_t agent_no):
        return self.thisptr.getAgentNumAgentNeighbors(agent_no)
    def getAgentNumObstacleNeighbors(self, size_t agent_no):
        return self.thisptr.getAgentNumObstacleNeighbors(agent_no)
    def getAgentNumORCALines(self, size_t agent_no):
        return self.thisptr.getAgentNumORCALines(agent_no)
    def getAgentObstacleNeighbor(self, size_t agent_no, size_t obstacle_no):
        return self.thisptr.getAgentObstacleNeighbor(agent_no, obstacle_no)
    def getAgentORCALine(self, size_t agent_no, size_t line_no):
        cdef Line line = self.thisptr.getAgentORCALine(agent_no, line_no)
        return line.point.x(), line.point.y(), line.direction.x(), line.direction.y()
    def getAgentPosition(self, size_t agent_no):
        cdef Vector2 pos = self.thisptr.getAgentPosition(agent_no)
        return pos.x(), pos.y()
    def getAgentPrefVelocity(self, size_t agent_no):
        cdef Vector2 velocity = self.thisptr.getAgentPrefVelocity(agent_no)
        return velocity.x(), velocity.y()
    def getAgentRadius(self, size_t agent_no):
        return self.thisptr.getAgentRadius(agent_no)
    def getAgentTimeHorizon(self, size_t agent_no):
        return self.thisptr.getAgentTimeHorizon(agent_no)
    def getAgentTimeHorizonObst(self, size_t agent_no):
        return self.thisptr.getAgentTimeHorizonObst(agent_no)
    def getAgentVelocity(self, size_t agent_no):
        cdef Vector2 velocity = self.thisptr.getAgentVelocity(agent_no)
        return velocity.x(), velocity.y()
    def getGlobalTime(self):
        return self.thisptr.getGlobalTime()
    def getNumAgents(self):
        return self.thisptr.getNumAgents()
    def getNumObstacleVertices(self):
        return self.thisptr.getNumObstacleVertices()
    def getObstacleVertex(self, size_t vertex_no):
        cdef Vector2 pos = self.thisptr.getObstacleVertex(vertex_no)
        return pos.x(), pos.y()
    def getNextObstacleVertexNo(self, size_t vertex_no):
        return self.thisptr.getNextObstacleVertexNo(vertex_no)
    def getPrevObstacleVertexNo(self, size_t vertex_no):
        return self.thisptr.getPrevObstacleVertexNo(vertex_no)
    def getTimeStep(self):
        return self.thisptr.getTimeStep()

    def processObstacles(self):
        with nogil:
            self.thisptr.processObstacles()

    def queryVisibility(self, tuple point1, tuple point2, float radius=0.0):
        cdef Vector2 c_point1 = Vector2(point1[0], point1[1])
        cdef Vector2 c_point2 = Vector2(point2[0], point2[1])
        cdef bool visible

        with nogil:
            visible = self.thisptr.queryVisibility(c_point1, c_point2, radius)

        return visible

    def setAgentDefaults(self, float neighbor_dist, size_t max_neighbors, float time_horizon,
                         float time_horizon_obst, float radius, float max_speed,
                         tuple velocity=(0, 0)):
        cdef Vector2 c_velocity = Vector2(velocity[0], velocity[1])
        self.thisptr.setAgentDefaults(neighbor_dist, max_neighbors, time_horizon,
                                      time_horizon_obst, radius, max_speed, c_velocity)

    def setAgentMaxNeighbors(self, size_t agent_no, size_t max_neighbors):
        self.thisptr.setAgentMaxNeighbors(agent_no, max_neighbors)
    def setAgentMaxSpeed(self, size_t agent_no, float max_speed):
        self.thisptr.setAgentMaxSpeed(agent_no, max_speed)
    def setAgentNeighborDist(self, size_t agent_no, float neighbor_dist):
        self.thisptr.setAgentNeighborDist(agent_no, neighbor_dist)
    def setAgentNeighborDist(self, size_t agent_no, float neighbor_dist):
        self.thisptr.setAgentNeighborDist(agent_no, neighbor_dist)
    def setAgentPosition(self, size_t agent_no, tuple position):
        cdef Vector2 c_pos = Vector2(position[0], position[1])
        self.thisptr.setAgentPosition(agent_no, c_pos)
    def setAgentPrefVelocity(self, size_t agent_no, tuple velocity):
        cdef Vector2 c_velocity = Vector2(velocity[0], velocity[1])
        self.thisptr.setAgentPrefVelocity(agent_no, c_velocity)
    def setAgentRadius(self, size_t agent_no, float radius):
        self.thisptr.setAgentRadius(agent_no, radius)
    def setAgentTimeHorizon(self, size_t agent_no, float time_horizon):
        self.thisptr.setAgentTimeHorizon(agent_no, time_horizon)
    def setAgentTimeHorizonObst(self, size_t agent_no, float timeHorizonObst):
        self.thisptr.setAgentTimeHorizonObst(agent_no, timeHorizonObst)
    def setAgentVelocity(self, size_t agent_no, tuple velocity):
        cdef Vector2 c_velocity = Vector2(velocity[0], velocity[1])
        self.thisptr.setAgentVelocity(agent_no, c_velocity)
    def setTimeStep(self, float time_step):
        self.thisptr.setTimeStep(time_step)
