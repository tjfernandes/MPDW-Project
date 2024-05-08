# Import custom modules
from config import CONFIG
import index_management as im
import search    


if __name__ == "__main__":
    index_name = CONFIG["user"]

    # #Get the OpenSearch client
    client = im.create_client()

    if not im.index_exists(client):
        print('\nCreating index:')
        print('Response index creation: \n', im.create_index(client))
        
        # Index the recipes
        print('\nIndexing recipes:')
        print('Response indexing: \n', im.add_recipes_to_index(client, im.recipes_data))
    else:
        print('\nIndex already exists')
    
    print("Welcome to this Recipe Helper!")
    while True:
        
        print("What would you like to do?")
        print("     1. Search for a recipe")
        print("     2. Delete the index and start over")
        print("     3. bla bla bla")
        print("     4. bla bla bla")
        print("     5. Exit")
        
        switch = {
            '1': lambda: (
                query := input('>> USER: '),
                search.text_query(client, index_name, query)
            ),
            '2': lambda: (
                print('\nDeleting index:'),
                print('Response index deletion: \n', im.delete_index(client, index_name)) 
            ),
            '3': lambda: print('3 Command'), # Do something else
            '4': lambda: print('4 Command'), # Do something else
            '5': lambda: exit()
        }

        choice = input('Choose an option: ')

        switch.get(choice, lambda: print('Invalid option'))()