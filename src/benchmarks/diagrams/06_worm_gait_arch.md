# Echo Architecture Diagram

This diagram visualizes the structure of the `PredictiveCodingGraph` and its internal components, highlighting the top-down hierarchical flow.

```mermaid
graph TD
    subgraph PCG["PredictiveCodingGraph"]
        direction TB
        JointHull["MarkovHull (Joint concatenated state)"]
        
        subgraph HTFF["HierarchicalThermoFlowFactor"]
            direction TB
            
            subgraph Macro["Macro Components"]
                direction TB
                macro_ebm["EBM (Gaussian or PrecisionWeighted)"]
                macro_hull["MarkovHull (Macro)"]
                macro_sol["SolenoidalFlow"]
                macro_diss["DissipativeFriction"]
                macro_therm["Thermostat"]
            end
            
            W_down["W_down (eqx.nn.Linear)"]
            
            subgraph Micro["Micro Components"]
                direction TB
                micro_ebm["EBM (Gaussian or PrecisionWeighted)"]
                micro_hull["MarkovHull (Micro)"]
                micro_sol["SolenoidalFlow"]
                micro_diss["DissipativeFriction"]
                micro_therm["Thermostat"]
            end
            
            macro_hull -->|Extracts Observable State| W_down
            macro_ebm -.->|Provides Precision - Stiffness| W_down
            W_down -->|Projects Belief| micro_hull
            micro_hull -.->|Prediction Error Gradient| macro_hull
        end
    end

    %% Styling
    classDef graphNode fill:#bbf,stroke:#333,stroke-width:2px;
    class PCG,HTFF graphNode;
```

### Explanation of the PredictiveCodingGraph

1. **HierarchicalThermoFlowFactor**: This is the core physics engine of the graph. It houses all of the primitive physics objects for both the micro and macro levels.
2. **Structural Coupling (`W_down`)**: The linear projection matrix that maps the macro state's predictions down into the micro level's latent space, allowing the two otherwise independent systems to interact via prediction errors.
3. **Top-Down Hierarchy**: The macro observer sits at the top of the hierarchy, generating beliefs that flow downward via `W_down` to constrain the micro observer, which forms the physical sensorimotor boundary of the agent.
