class Manager (object):
    """
    Module manager.

    """
    
    def __init__(self):
        self.module_list = []
        self.conn_list = []

    def add_module(self, module):
        """
        Add a module instance to the simulation.
        """
        
        self.module_list.append(module)

    def rm_module(self, module):
        """
        Remove a module instance from the simulation.
        """
        
        self.module_list.remove(module)

    def add_connectivity(self, conn):
        """
        Add a connectivity object to the simulation.
        """
        
        self.conn_list.append(conn)

    def rm_connectivity(self, conn):
        """
        Remove a connectivity object from the simulation.
        """
        
        self.conn_list.remove(conn)

    def connect(self, m1, m2, conn):
        """
        Connect two modules using the specified connectivity object.
        """

        pass

    def connect_all(self):
        """
        Connect all modules using their associated connectivity objects.
        """

        pass
    
    def start(self):
        """
        Start all of the module processes.
        """
        
        for m in self.module_list:
            m.start()
