# System Architecture

### Hybrid Network Architecture

This diagram illustrates the data flow inside `UniversalBlackjackHybridPolicyNetwork`, highlighting the input transformation and the quantum-classical interface.

```mermaid
flowchart TD
    subgraph "Classical Pre-processing"
        Input([Blackjack State]) -->|3 or 45 features| Encoder[Classical Encoder]
        Encoder -->|"Linear (No Activation)"| Compression[Compression Layer]
    end

    subgraph "Quantum Circuit (PennyLane)"
        Compression -->|Unbounded Input| Transform{{"Input Transform<br>(arctan * scale)"}}
        Transform -->|"Bounded Angles [-π, π]"| QEmbed[Quantum Embedding]
        
        subgraph "QNode"
            QEmbed -->|RY / RZ Gates| QLayers[Variational Layers]
            QLayers -->|Entanglement| QLayers
            QLayers -->|Measurement| QMeasure[Pauli-Z / Amplitude]
        end
    end

    subgraph "Classical Post-processing"
        QMeasure -->|Quantum Features| PostProc[Post-processing Layer]
        PostProc -->|Linear| Softmax[Softmax]
        Softmax --> Output([Action Probabilities])
    end

    style Transform fill:#f96,stroke:#333,stroke-width:2px
    style QNode fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,stroke-dasharray: 5 5
```

## System Overview (A2C Training Loop)

This diagram shows how the `A2CAgent` interacts with the environment and how the hybrid network fits into the broader reinforcement learning context.

```mermaid
sequenceDiagram
    participant Env as Blackjack Environment
    participant Agent as A2C Agent
    participant PolicyNet as "Hybrid Policy Net (Actor)"
    participant Critic as "Value Net (Critic)"

    loop Training Episode
        Env->>Agent: State (Player Sum, Dealer Card, Ace)
        
        rect rgb(240, 248, 255)
            Note over Agent, PolicyNet: Forward Pass
            Agent->>PolicyNet: Get Action Probabilities
            PolicyNet-->>Agent: Probs [Stand, Hit]
            Agent->>Agent: Sample Action
        end
        
        Agent->>Env: Execute Action
        Env-->>Agent: Reward, Next State, Done
        
        opt Every N Steps
            rect rgb(255, 240, 240)
                Note over Agent, Critic: Optimization Step
                Agent->>Critic: Estimate Value V(s)
                Agent->>Agent: Calculate Advantage A = R - V(s)
                Agent->>PolicyNet: Update Weights (Policy Loss)
                Agent->>Critic: Update Weights (Value Loss)
            end
        end
    end
```
