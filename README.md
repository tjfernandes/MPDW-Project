
# WSDM Project

This project allows users to ask general information about some recipe. They can do it by engaging in a dialog with the system. Thee user can ask for recipes, select the one they want an ask information during the instructions provided by the system for each recipe.

The user should run the program in an environment with the necessary libraries.

Start the app by running:

```bash
python app.py
```

You are then prompted with multiple options:

| Option | Action   | Description                          |
| :----- | :---------- :------------------------------------ |
| 1      | `Search for a recipe`   | Searches for recipes using OpenSearch |
| 2      | `Search for an image using text`   | Searches for an image related to the recipe query from the user |
| 3      | `Search for text using image`   | Searches for a recipe related to the image query from the user |
| 4      | `Search for image using image`   | Searches for a image related to the image query from the user |
| 5      | `Make PlanLLM request`   | Start PlanLLM dialog related to some recipe (for testing) |
| 6      | `Continue PlanLLM requests`   | Continues PlanLLM dialog related to some recipe (for testing) |
| 7      | `Start Dialog`   | Start a dialog using the dialog manager (this is the most complex operation makes use of all the previous ones) |
| 8      | `Delete the index and start over`   | Deletes OpenSearch index, re-creates it and indexes all the recipes again|
| 9      | `Exit`   | Exit the application |




When starting the dialog, the user can ask for some recipes and the system provides a set of 3 related recipes. If the user wants more recipes, they can ask for more recipes and the system provides 3 more.