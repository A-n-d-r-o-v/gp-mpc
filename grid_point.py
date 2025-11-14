
class GridPoint():
    def __init__(self, initialState, finalState, name, ID, group):
        self.ID = ID
        self.group = group
        self.initialState = initialState
        self.finalState = finalState
        self.trajectory = None # UNUSED
        self.name = name
        self.cost = 0