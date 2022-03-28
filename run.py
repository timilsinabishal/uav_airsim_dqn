from ddqn_agent import DDQN_Agent

if __name__ == "__main__":
    ddqn_agent = DDQN_Agent(useDepth=True)
    ddqn_agent.train()
