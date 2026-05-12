import json
from chan_client import ChanClient

client = ChanClient()

board = "mu"
catalog = client.get_catalog(board)

posts = []
max_threads = 10

for page in catalog:
    for thread in page["threads"][:max_threads]:
        thread_number = thread["no"]
        thread_data = client.get_thread(board, thread_number)

        for post in thread_data.get("posts", []):
            posts.append(post)

print(f"Collected {len(posts)} posts from /mu/")

with open("music_posts.json", "w", encoding="utf-8") as file:
    json.dump(posts, file, ensure_ascii=False, indent=2)

print("Saved posts to music_posts.json")