# Import custom modules
from config import CONFIG
import index_management as im
import search    

def init_index(client):
    # Create the index
    print('\nCreating index:')
    print('Response index creation: \n', im.create_index(client))
    
    # Index the recipes
    print('\nIndexing recipes:')
    print('Response indexing: \n', im.add_recipes_to_index(client, im.recipes_data))

if __name__ == "__main__":
    index_name = CONFIG["user"]

    # #Get the OpenSearch client
    client = im.create_client()

    if not im.index_exists(client):
        init_index(client)
        
    
    print("Welcome to this Recipe Helper!")
    while True:
        
        print("What would you like to do?")
        print("     1. Search for a recipe")
        print("     2. Search for an image using text")
        print("     3. Search for text using image")
        print("     4. Search for image using image")
        print("     5. Delete the index and start over")
        print("     6. Exit")
        
        switch = {
            '1': lambda: (
                query := input('>> USER:    '),
                search.text_query(client, index_name, query)
            ),
            '2': lambda: (
                query := input('>> USER (text-to-image):    '),
                search.text_to_image(client, index_name, query)
            ),
            '3': lambda: (
                url := input('>> USER (image-to-text):    '),
                search.image_to_text(client, index_name, url)
            ),
            '4': lambda: (
                url := input('>> USER (image-to-image):    '),
                search.image_to_image(client, index_name, url)
            ),
            '5': lambda: (
                print('\nDeleting index:'),
                print('Response index deletion: \n', im.delete_index(client, index_name)),
                init_index(client) 
            ), # Do something else
            '6': lambda: exit()
        }

        choice = input('Choose an option: ')

        switch.get(choice, lambda: print('Invalid option'))()