Reference from here - 'https://robosuite.ai/docs/modules/robots.html'

1. During initilization of environment (suite.make(....)), individual robots are both instantiated and initialized

2. During a given simulation call (env.step(...)), the environment will receive a set of actions and distribute them accordingly to each robot, according to their respective action spaces. Each robot then converts these actions into low-level torques via their respective controllers, and directly executes these torques in the simulation

Now for the Robosuite Objects - 'https://robosuite.ai/docs/modules/objects.html'