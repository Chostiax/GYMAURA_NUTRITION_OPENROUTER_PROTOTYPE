TEST_CASES = [
              
        {
            "input": "I had some coffee and eggs for breakfast",
            "expected": [
                {"food": "coffee", "quantity": None},
                {"food": "egg", "quantity": None},
            ],
        },
        {
            "input": "My mom made us an apple pie for the party",
            "expected": [
                {"food": "apple pie", "quantity": None},
            ],
        },
        {
            "input": "I ate some pasta and drank a can of coke",
            "expected": [
                {"food": "pasta", "quantity": None},
                {"food": "coke", "quantity": 1},
            ],
        },
        {
            "input": "For dinner, I had a pepperoni pizza",
            "expected": [
                {"food": "pepperoni pizza", "quantity": 1},
            ],
        },
        {
            "input": "My girlfriend baked some cookies for a snack",
            "expected": [
                {"food": "cookie", "quantity": None},
            ],
        },
        {
            "input": "I had broccoli and carrots with chicken breast for lunch",
            "expected": [
                {"food": "broccoli", "quantity": None},
                {"food": "carrot", "quantity": None},
                {"food": "chicken", "quantity": None},
            ],
        },
        {
            "input": "We went to an Italian restaurant and I ordered lasagna with tiramisu for dessert",
            "expected": [
                {"food": "lasagna", "quantity": 1},
                {"food": "tiramisu", "quantity": 1},
            ],
        },
        {
            "input": "I wasn’t feeling hungry, so I just ate some dates and almonds",
            "expected": [
                {"food": "date", "quantity": None},
                {"food": "almond", "quantity": None},
            ],
        },
        {
            "input": "I ate a fruit salad with 2 cups of orange juice, an apple, 4 strawberries, a banana, and a peach",
            "expected": [
                {"food": "fruit salad", "quantity": None},
                {"food": "orange juice", "quantity": 2},
                {"food": "apple", "quantity": 1},
                {"food": "strawberry", "quantity": 4},
                {"food": "banana", "quantity": 1},
                {"food": "peach", "quantity": 1},
            ],
        },
        {
            "input": "For breakfast, I had a tuna omelet with a glass of orange juice",
            "expected": [
                {"food": "tuna omelet", "quantity": 1},
                {"food": "orange juice", "quantity": 1},
            ],
        },
        {
            "input": "I didn’t eat anything yet today",
            "expected": [],
        },
        {
            "input": "This afternoon I had a blueberry muffin and a latte",
            "expected": [
                {"food": "blueberry muffin", "quantity": 1},
                {"food": "latte", "quantity": 1},
            ],
        },
        {
            "input": "We ordered takeout and I got green curry with chicken and jasmine rice",
            "expected": [
                {"food": "green curry", "quantity": None},
                {"food": "chicken", "quantity": None},
                {"food": "jasmine rice", "quantity": None},
            ],
        },
        {
            "input": "I just had a protein shake after my workout",
            "expected": [
                {"food": "protein shake", "quantity": 1},
            ],
        },
        {
            "input": "For lunch, I had quinoa with roasted vegetables and grilled salmon",
            "expected": [
                {"food": "quinoa", "quantity": None},
                {"food": "vegetable", "quantity": None},
                {"food": "salmon", "quantity": None},
            ],
        },
        {
            "input": "I drank about three cups of coffee so far today",
            "expected": [
                {"food": "coffee", "quantity": 3},
            ],
        },
        {
            "input": "Later I had a steak and a scoop of potato salad",
            "expected": [
                {"food": "steak", "quantity": 1},
                {"food": "potato salad", "quantity": 1},
            ],
        },
        {
            "input": "I skipped breakfast this morning, just had a black coffee",
            "expected": [
                {"food": "black coffee", "quantity": 1},
            ],
        },
        {
            "input": "For a snack, I had a cup of Greek yogurt with some blueberries",
            "expected": [
                {"food": "yogurt", "quantity": 1},
                {"food": "blueberry", "quantity": None},
            ],
        },
        {
            "input": "Last night I finished the leftovers: half a plate of spaghetti bolognese",
            "expected": [
                {"food": "spaghetti bolognese", "quantity": 0.5},
            ],
        },
        {
            "input": "I grabbed a chicken caesar wrap and a bottle of iced tea for lunch",
            "expected": [
                {"food": "chicken caesar wrap", "quantity": 1},
                {"food": "iced tea", "quantity": 1},
            ],
        },
        {
            "input": "For dessert, I had a slice of New York cheesecake",
            "expected": [
                {"food": "cheesecake", "quantity": 1},
            ],
        },
        {
            "input": "I was snacking on popcorn while watching a movie",
            "expected": [
                {"food": "popcorn", "quantity": None},
            ],
        },
        {
            "input": "My breakfast was simple, just a bowl of cereal with milk",
            "expected": [
                {"food": "cereal", "quantity": 1},
                {"food": "milk", "quantity": None},
            ],
        },
        {
            "input": "I think I drank two cappuccinos this morning",
            "expected": [
                {"food": "cappuccino", "quantity": 2},
            ],
        },
        {
            "input": "I made a smoothie with spinach, a banana, and almond milk",
            "expected": [
                {"food": "smoothie", "quantity": 1},
                {"food": "spinach", "quantity": None},
                {"food": "banana", "quantity": 1},
                {"food": "almond milk", "quantity": None},
            ],
        },
        {
            "input": "I had ribeye with a baked potato and asparagus",
            "expected": [
                {"food": "ribeye", "quantity": None},
                {"food": "potato", "quantity": 1},
                {"food": "asparagus", "quantity": None},
            ],
        },
        {
            "input": "I ate some sweet potato with grilled chicken",
            "expected": [
                {"food": "sweet potato", "quantity": None},
                {"food": "chicken", "quantity": None},
            ],
        },
        {
            "input": "I ate a bag of chips and some tomato salsa while waiting for dinner",
            "expected": [
                {"food": "chips", "quantity": 1},
                {"food": "tomato salsa", "quantity": None},
            ],
        },
    ]
