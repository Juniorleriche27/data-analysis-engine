# Meta Model API Integration

## Provider

- Base URL: `https://api.meta.ai/v1`
- Model: `muse-spark-1.3`
- Authentication environment variable: `MODEL_API_KEY`
- Planning transport: Chat Completions
- Planning output: JSON Schema structured output

## Architecture

Meta Muse Spark is the reasoning and planning layer.

It may:

- interpret a user's analytical request;
- inspect dataset metadata and profiling context;
- choose native engine methods;
- choose configuration parameters;
- interpret deterministic engine outputs.

It may not:

- calculate native analytical results itself;
- inject a result into an analysis call;
- fabricate raw dataset values;
- bypass the native engine.

The planner therefore outputs method selection and configuration only.

A later binding stage attaches the real dataset to the validated plan before deterministic engine execution.

## Secret handling

`MODEL_API_KEY` must stay in the environment and must never be committed to Git.
