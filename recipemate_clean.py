
import os
import re
import pickle
import statistics
import faiss

from collections import Counter
from sentence_transformers import SentenceTransformer

chat_history = {
    "last_search_results": [],
    "selected_recipe_id": None,
    "last_user_ingredients": [],
    "last_restrictions": []
}

units = [
    "tablespoons", "tablespoon", "tbsp",
    "teaspoons", "teaspoon", "tsp",
    "cups", "cup", "ounces", "ounce", "oz",
    "pounds", "pound", "lbs", "lb",
    "grams", "gram", "g", "kilograms", "kilogram", "kg",
    "milliliters", "milliliter", "ml",
    "liters", "liter", "l", "cloves", "clove"
]

containers = [
    "cans", "can", "packages", "package",
    "jars", "jar", "bottles", "bottle",
    "boxes", "box"
]

unit_conversions = {
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tbsp": "tbsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tsp": "tsp",
    "cups": "cup",
    "cup": "cup",
    "pounds": "lb",
    "pound": "lb",
    "lbs": "lb",
    "lb": "lb",
    "ounces": "oz",
    "ounce": "oz",
    "oz": "oz",
    "cloves": "clove",
    "clove": "clove"
}


def semantic_search(query, limit=5):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, limit)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            recipe_id = recipe_ids[idx]
            recipe = recipes.get(recipe_id)

            if recipe:
                results.append({
                    "id": recipe_id,
                    "title": recipe["title"],
                    "score": float(score)
                })

    return results


def normalize_ingredient(ingredient):

    if not isinstance(ingredient, str):
        return ""

    text = ingredient.lower()

    text = re.sub(r'advertisement', '', text)
    text = re.sub(r'\([^)]*\)', '', text)

    text = re.sub(r'^\s*\d+\s+\d+/\d+\s*', '', text)
    text = re.sub(r'^\s*\d+/\d+\s*', '', text)
    text = re.sub(r'^\s*\d+(?:\.\d+)?\s*', '', text)

    text = re.sub(r'\b(skinless)\b', '', text)

    preparation_words = [
        "finely diced", "roughly chopped",
        "chopped", "minced", "diced",
        "sliced", "grated", "shredded",
        "torn into pieces"
    ]

    for word in preparation_words:
        text = text.replace(word, "")

    for unit in sorted(units, key=len, reverse=True):
        text = re.sub(rf'\b{unit}\b', '', text)

    for container in containers:
        text = re.sub(rf'\b{container}\b', '', text)

    text = re.sub(r'\bof\b', '', text)
    text = re.sub(r'[^a-z\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def search_recipes(query, limit=5):
    query = query.lower().strip()

    stop_words = {
        "a", "an", "the", "is", "are", "for", "with",
        "and", "or", "to", "of", "in", "on", "recipe",
        "recipes", "make", "find", "show", "me"
    }

    query_words = {
        word for word in re.findall(r'\b[a-z]+\b', query)
        if word not in stop_words
    }

    if not query_words:
        return []

    results = []

    for recipe in recipes.values():
        title = recipe.get("title") or ""
        ingredients = recipe.get("normalized_ingredients") or []

        title_words = set(re.findall(r'\b[a-z]+\b', title.lower()))

        matched_title_words = query_words.intersection(title_words)

        matched_ingredients = [
            ingredient
            for ingredient in ingredients
            if ingredient and any(
                word in ingredient.split()
                for word in query_words
            )
        ]

        matched_ingredient_words = set()

        for ingredient in ingredients:
            ingredient_words = set(ingredient.split())
            matched_ingredient_words.update(
                query_words.intersection(ingredient_words)
            )

        score = (
            5 * len(matched_title_words)
            + len(matched_ingredient_words)
        )

        if query in title.lower():
            score += 10

        if query_words.issubset(title_words):
            score += 5

        if score > 0:
            results.append({
                "id": recipe["id"],
                "title": title,
                "score": score,
                "matched_title_words": sorted(matched_title_words),
                "matched_ingredient_words": sorted(matched_ingredient_words),
                "matched_ingredients": matched_ingredients,
                "ingredients": recipe.get("ingredients") or [],
                "instructions": recipe.get("instructions") or ""
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:limit]


def hybrid_search(query, limit=5):

    semantic_results = semantic_search(query, limit=20)
    keyword_results = search_recipes(query, limit=20)

    combined = {}

    for recipe in semantic_results:
        combined[recipe["id"]] = {
            "id": recipe["id"],
            "title": recipe["title"],
            "semantic_score": recipe["score"],
            "keyword_score": 0
        }

    for recipe in keyword_results:
        recipe_id = recipe["id"]

        if recipe_id not in combined:
            combined[recipe_id] = {
                "id": recipe_id,
                "title": recipe["title"],
                "semantic_score": 0,
                "keyword_score": 0
            }

        combined[recipe_id]["keyword_score"] = recipe["score"]

    results = list(combined.values())

    for recipe in results:
        recipe["final_score"] = (
            0.7 * recipe["semantic_score"]
            + 0.3 * (
                recipe["keyword_score"] /
                max([r["keyword_score"] for r in results] + [1])
            )
        )

    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results[:limit]


def match_recipes_by_ingredients(
    user_ingredients,
    limit=5,
    restrictions=None
):
    user_ingredients = list(dict.fromkeys(
        normalize_ingredient(item)
        for item in user_ingredients
        if normalize_ingredient(item)
    ))

    results = []

    if not user_ingredients:
        return []

    for recipe in recipes.values():

        filtered = filter_restricted_recipes(
            [recipe],
            restrictions or []
        )

        if not filtered:
            continue

        recipe_ingredients = [
            item for item in recipe.get("normalized_ingredients", [])
            if item
        ]

        matched = []
        missing = []

        for ingredient in user_ingredients:
            found = any(
                ingredient_matches(ingredient, item)
                for item in recipe_ingredients
            )

            if found:
                matched.append(ingredient)
            else:
                missing.append(ingredient)

        if not matched:
            continue

        additional = []

        for item in recipe_ingredients:
            found = any(
                ingredient_matches(user_item, item)
                for user_item in user_ingredients
            )

            if not found:
                additional.append(item)

        match_score = len(matched) / len(user_ingredients)
        coverage = len(matched) / max(len(recipe_ingredients), 1)

        final_score = 0.7 * match_score + 0.3 * coverage

        results.append({
            "id": recipe["id"],
            "title": recipe.get("title"),
            "matched_ingredients": matched,
            "missing_ingredients": missing,
            "additional_ingredients": list(dict.fromkeys(additional)),
            "match_score": round(match_score, 2),
            "coverage": round(coverage, 2),
            "final_score": round(final_score, 3),
            "ingredients": recipe.get("ingredients") or []
        })

    results.sort(
        key=lambda x: (
            x["match_score"],
            -len(x["additional_ingredients"]),
            x["final_score"]
        ),
        reverse=True
    )

    return results[:limit]


def extract_quantity(ingredient):
    if not isinstance(ingredient, str):
        return None, None

    ingredient = ingredient.strip()

    match = re.match(
        r'^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)'
        r'(?:\s+(tablespoons?|tbsp|teaspoons?|tsp|cups?|'
        r'pounds?|lbs?|ounces?|oz|cloves?|grams?|g|'
        r'kilograms?|kg|milliliters?|ml|liters?|l))?\b',
        ingredient,
        flags=re.IGNORECASE
    )

    if not match:
        return None, None

    quantity = match.group(1)
    unit = match.group(2)

    if " " in quantity:
        whole, fraction = quantity.split()
        numerator, denominator = fraction.split("/")
        quantity = float(whole) + float(numerator) / float(denominator)
    elif "/" in quantity:
        numerator, denominator = quantity.split("/")
        quantity = float(numerator) / float(denominator)
    else:
        quantity = float(quantity)

    return quantity, unit


def standardize_unit(unit):
    if not unit:
        return None

    return unit_conversions.get(unit.lower(), unit.lower())


def compare_quantities(user_quantity, recipe_quantity,
                       user_unit, recipe_unit):

    if user_quantity is None or recipe_quantity is None:
        return "Quantity unknown"

    user_unit = standardize_unit(user_unit)
    recipe_unit = standardize_unit(recipe_unit)

    if user_unit is None and recipe_unit is None:
        return "Sufficient" if user_quantity >= recipe_quantity else "Insufficient"

    if user_unit is None or recipe_unit is None:
        return "Quantity unknown"

    if user_unit != recipe_unit:
        return "Quantity unknown"

    if user_quantity >= recipe_quantity:
        return "Sufficient"

    return "Insufficient"


def ingredient_matches(user_item, recipe_item):
    user_words = set(user_item.split())
    recipe_words = set(recipe_item.split())

    if not user_words or not recipe_words:
        return False

    exclusions = {
        "chicken": ["broth", "stock", "bouillon"],
        "garlic": ["salt", "powder"],
        "onion": ["powder"],
        "tomato": ["sauce", "soup", "paste"]
    }

    for ingredient, excluded in exclusions.items():
        if ingredient in user_words:
            if any(word in recipe_words for word in excluded):
                return False

    variations = {
        "tomato": ["tomato", "tomatoes"],
        "potato": ["potato", "potatoes"]
    }

    for word in user_words:
        allowed = variations.get(word, [word])

        if not any(item in recipe_words for item in allowed):
            return False

    return True


def parse_ingredient(ingredient):
    if not isinstance(ingredient, str):
        return None

    ingredient = re.sub(
        r'advertisement', '', ingredient, flags=re.IGNORECASE
    ).strip()

    if not ingredient:
        return None

    quantity, unit = extract_quantity(ingredient)

    remaining = re.sub(
        r'^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*',
        '',
        ingredient
    ) if quantity is not None else ingredient

    if unit:
        remaining = re.sub(
            rf'^\s*{re.escape(unit)}\b',
            '',
            remaining,
            flags=re.IGNORECASE
        )

    name = normalize_ingredient(remaining)

    if not name:
        return None

    return {
        "name": name,
        "quantity": quantity,
        "unit": standardize_unit(unit)
    }


def check_recipe_ingredients(user_ingredients, recipe_ingredients):
    user_data = [
        parse_ingredient(item) for item in user_ingredients
    ]

    recipe_data = [
        parse_ingredient(item) for item in recipe_ingredients
    ]

    user_data = [item for item in user_data if item]
    recipe_data = [item for item in recipe_data if item]

    results = []

    for required in recipe_data:
        available = next(
            (
                item for item in user_data
                if ingredient_matches(
                    item["name"], required["name"]
                )
            ),
            None
        )

        if available is None:
            status = "Not available"

        elif (
            available["quantity"] is None
            or required["quantity"] is None
        ):
            status = "Quantity unknown"

        else:
            status = compare_quantities(
                available["quantity"],
                required["quantity"],
                available["unit"],
                required["unit"]
            )

        results.append({
            "ingredient": required["name"],
            "required": required["quantity"],
            "unit": required["unit"],
            "status": status
        })

    return results


def summarize_ingredient_status(comparison):

    sufficient = sum(
        item["status"] == "Sufficient" for item in comparison
    )

    insufficient = sum(
        item["status"] == "Insufficient" for item in comparison
    )

    unavailable = sum(
        item["status"] == "Not available" for item in comparison
    )

    unknown = sum(
        item["status"] in ["Quantity unknown", "Unit mismatch"]
        for item in comparison
    )

    return {
        "sufficient": sufficient,
        "insufficient": insufficient,
        "unavailable": unavailable,
        "unknown": unknown,
        "can_prepare": (
            insufficient == 0
            and unavailable == 0
            and unknown == 0
        )
    }


def recommend_recipes_with_quantities(
    user_ingredients,
    limit=5,
    restrictions=None
):
    results = match_recipes_by_ingredients(
        user_ingredients,
        limit=limit,
        restrictions=restrictions
    )

    recommendations = []

    for result in results:
        recipe = recipes[result["id"]]

        comparison = check_recipe_ingredients(
            user_ingredients,
            recipe.get("ingredients") or []
        )

        summary = summarize_ingredient_status(comparison)

        sufficient = [
            item["ingredient"]
            for item in comparison
            if item["status"] == "Sufficient"
        ]

        insufficient = [
            item["ingredient"]
            for item in comparison
            if item["status"] == "Insufficient"
        ]

        unavailable = [
            item["ingredient"]
            for item in comparison
            if item["status"] == "Not available"
        ]

        unknown = [
            item["ingredient"]
            for item in comparison
            if item["status"] == "Quantity unknown"
        ]

        recommendations.append({
            "id": result["id"],
            "title": result["title"],
            "match_score": result["match_score"],
            "matched_ingredients": result["matched_ingredients"],
            "sufficient_ingredients": sufficient,
            "insufficient_ingredients": insufficient,
            "unavailable_ingredients": unavailable,
            "unknown_quantity_ingredients": unknown,
            "ingredient_status": comparison,
            "summary": summary
        })

    recommendations.sort(
        key=lambda x: (
            x["summary"]["sufficient"],
            -x["summary"]["insufficient"],
            -x["summary"]["unavailable"],
            x["match_score"]
        ),
        reverse=True
    )

    return recommendations


def is_vegetarian(recipe):
    non_vegetarian_ingredients = [
        "chicken", "beef", "pork", "mutton", "lamb",
        "fish", "salmon", "tuna", "shrimp", "prawn",
        "crab", "lobster", "turkey", "bacon", "ham",
        "sausage", "anchovy", "meat", "duck"
    ]

    title = (recipe.get("title") or "").lower()

    title_keywords = [
        "pot roast", "fisherman", "fish pie"
    ]

    if any(word in title for word in title_keywords):
        return False

    ingredients = recipe.get("normalized_ingredients") or []
    text = " ".join(
        str(item).lower() for item in ingredients if item
    )

    words = set(text.split())

    for item in non_vegetarian_ingredients:
        if item in words:
            return False

    return True


def filter_vegetarian_recipes():
    vegetarian_recipes = []

    for recipe in recipes.values():
        if is_vegetarian(recipe):
            vegetarian_recipes.append(recipe)

    return vegetarian_recipes


def filter_restricted_recipes(candidate_recipes, restrictions):
    restriction_keywords = {
        "peanuts": [
            "peanut", "peanuts", "peanut oil",
            "peanut butter", "groundnut"
        ],
        "dairy": [
            "milk", "buttermilk", "cheese", "butter",
            "cream", "yogurt", "curd", "ghee",
            "whey", "casein", "milk powder",
            "condensed milk", "evaporated milk"
        ],
        "eggs": ["egg", "eggs", "egg yolk", "egg white"],
        "chicken": ["chicken"],
        "meat": [
            "beef", "pork", "mutton", "lamb",
            "chicken", "turkey"
        ]
    }

    filtered = []

    for recipe in candidate_recipes:
        ingredients = recipe.get("normalized_ingredients") or []

        text = " ".join(
            str(item).lower() for item in ingredients if item
        )

        excluded = False

        for restriction in restrictions:
            keywords = restriction_keywords.get(
                restriction.lower(), [restriction.lower()]
            )

            if any(keyword in text for keyword in keywords):
                excluded = True
                break

        if not excluded:
            filtered.append(recipe)

    return filtered


def find_substitutions(target_ingredient):
    results = []

    for recipe_id, recipe in substitution_index.items():
        text = recipe["instructions"]

        pattern = (
            r'[^.!?]*\bsubstitut\w*\b[^.!?]*'
            r'\b' + re.escape(target_ingredient) +
            r'\b[^.!?]*[.!?]?'
        )

        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append({
                "id": recipe_id,
                "title": recipe["title"],
                "evidence": match.group().strip()
            })

    unique_results = {
        (item["id"], item["evidence"]): item
        for item in results
    }

    return list(unique_results.values())


def extract_substitute(evidence, target_ingredient):
    patterns = [
        r'\bsubstitut\w*\s+(.+?)\s+for\s+' + re.escape(target_ingredient) + r'\b',
        r'\b(.+?)\s+in place of\s+' + re.escape(target_ingredient) + r'\b',
        r'\buse\s+(.+?)\s+instead of\s+' + re.escape(target_ingredient) + r'\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, evidence, re.IGNORECASE)

        if match:
            substitute = match.group(1).strip()
            return {
                "original_ingredient": target_ingredient,
                "substitute_ingredient": substitute
            }

    return None


def extract_substitution_quantity(instructions, target_ingredient):
    pattern = (
        r'(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)'
        r'\s*(tablespoons?|tbsp|teaspoons?|tsp|cups?)'
        r'\s+oil\b'
    )

    match = re.search(pattern, instructions, re.IGNORECASE)

    if not match:
        return None

    return {
        "quantity": match.group(1),
        "unit": match.group(2)
    }


def get_substitution_recommendations(target_ingredient):
    candidates = find_substitutions(target_ingredient)
    recommendations = []

    for item in candidates:
        substitute = extract_substitute(
            item["evidence"], target_ingredient
        )

        if not substitute:
            continue

        recipe = recipes.get(item["id"])
        if not recipe:
            continue

        quantity = extract_substitution_quantity(
            recipe.get("instructions") or "",
            target_ingredient
        )

        recommendations.append({
            "recipe_id": item["id"],
            "recipe_title": item["title"],
            "original_ingredient": target_ingredient,
            "substitute_ingredient": substitute["substitute_ingredient"],
            "suggested_quantity": quantity,
            "evidence": item["evidence"]
        })

    return recommendations


def chatbot_response(user_message):
    message = user_message.lower().strip()

    if not message:
        return "Please enter a message."

    # Statistics and analytical queries
    if any(word in message for word in [
        "how many recipes", "total recipes", "common ingredients",
        "most used ingredients", "commonly used ingredients",
        "average", "median", "most steps", "instruction steps"
    ]):
        return statistics_response(message)

    # Contextual questions about selected recipe
    if any(word in message for word in [
        "how long", "how much time", "how many minutes",
        "how many hours", "what temperature"
    ]):
        return answer_recipe_question(message)

    # Follow-up questions about previous search results
    follow_up_phrases = [
        "first recipe", "second recipe", "third recipe",
        "first one", "second one", "third one",
        "show ingredients", "show instructions",
        "tell me more", "details of"
    ]

    if any(phrase in message for phrase in follow_up_phrases):
        return handle_follow_up(message)

    if any(phrase in message for phrase in [
        "which recipe uses",
        "which recipes use",
        "which recipe contains",
        "which recipes contain",
        "among these",
        "among the previous"
    ]):
        return check_previous_recipe_ingredients(message)

    # Ingredient substitutions
        # Substitution
    if any(word in message for word in ["substitute", "replacement", "instead of", "replace"]):

        if "this recipe" in message or "in this recipe" in message:
            match = re.search(
                r"replace\s+(.+?)\s+with\s+(.+?)(?:\s+in this recipe|\?|$)",
                message
            )

            if match:
                target = match.group(1).strip()
                replacement = match.group(2).strip()
                return recipe_specific_substitution(target, replacement)

        target = re.search(
            r"(?:instead of|replace|substitute)\s+(.+?)(?:\?|$)",
            message
        )

        if target:
            return substitution_response(target.group(1).strip())

        return "Which ingredient would you like to replace?"

    # Ingredient-based recipe matching
    if "what can i make with" in message or "ingredients i have" in message:
        ingredients = message.replace("what can i make with", "")
        ingredients = ingredients.replace("ingredients i have", "")

        restrictions = []

        restriction_match = re.search(
            r"\bwithout\s+(.+?)(?:\?|$)",
            ingredients
        )

        if restriction_match:
            restrictions = [
                item.strip()
                for item in restriction_match.group(1).split(",")
            ]
            ingredients = ingredients[:restriction_match.start()]

        ingredients = re.sub(r"\band\b", ",", ingredients)

        user_ingredients = [
            item.strip()
            for item in ingredients.split(",")
            if item.strip()
        ]

        chat_history["last_user_ingredients"] = user_ingredients
        chat_history["last_restrictions"] = restrictions

        return ingredient_match_response(
            user_ingredients,
            restrictions=restrictions
        )

    # Default recipe search
    return recipe_search_response(user_message)


def recipe_search_response(query):
    results = hybrid_search(query, limit=5)

    chat_history["last_search_results"] = results
    chat_history["selected_recipe_id"] = None

    if not results:
        return "I couldn't find any matching recipes in the dataset."

    response = "Here are some recipes I found:\n\n"

    for i, item in enumerate(results, 1):
        response += f"{i}. {item['title']}\n"
        response += f"   Recipe ID: {item['id']}\n\n"

    return response


def handle_follow_up(message):
    message = message.lower().strip()
    results = chat_history["last_search_results"]

    if not results:
        return "Please search for recipes first."

    selected_recipe = None

    for recipe in results:
        title = recipe["title"].lower()
        if title in message or message in title:
            selected_recipe = recipe
            break

    if selected_recipe is None:
        if any(word in message for word in ["first", "1st"]):
            selected_recipe = results[0]
        elif any(word in message for word in ["second", "2nd"]):
            if len(results) < 2:
                return "There is no second recipe in the previous results."
            selected_recipe = results[1]
        elif any(word in message for word in ["third", "3rd"]):
            if len(results) < 3:
                return "There is no third recipe in the previous results."
            selected_recipe = results[2]
        else:
            return "I couldn't identify that recipe in the previous results."

    chat_history["selected_recipe_id"] = selected_recipe["id"]
    return get_recipe_details(selected_recipe["id"])


def check_previous_recipe_ingredients(message):
    results = chat_history["last_search_results"]

    if not results:
        return "Please search for recipes first."

    ingredient = re.search(
        r"(?:uses|use|contains|contain|have|has)\s+(.+?)(?:\s+among\b|\?|$)",
        message.lower()
    )

    if not ingredient:
        return "Which ingredient would you like me to check?"

    target = ingredient.group(1).strip()
    response = f"Recipes containing {target}:\n\n"
    found = []

    for recipe in results:
        full_recipe = recipes.get(recipe["id"])

        if not full_recipe:
            continue

        ingredients = full_recipe.get("normalized_ingredients") or []

        if any(target in item for item in ingredients):
            found.append(recipe)

    if not found:
        return f"None of the previous {len(results)} recipes contain {target}."

    for i, recipe in enumerate(found, 1):
        response += f"{i}. {recipe['title']}\n"
        response += f"   Recipe ID: {recipe['id']}\n\n"

    return response


def get_recipe_details(recipe_id):
    recipe = recipes.get(recipe_id)

    if not recipe:
        return "Recipe not found."

    response = f"Recipe: {recipe['title']}\n\n"
    response += "Ingredients:\n"

    for ingredient in recipe.get("ingredients") or []:
        response += f"- {ingredient}\n"

    response += "\nInstructions:\n"
    response += str(recipe.get("instructions") or "Not available")

    return response


def ingredient_match_response(user_ingredients, restrictions=None):
    results = match_recipes_by_ingredients(
        user_ingredients,
        limit=5,
        restrictions=restrictions
    )

    if not results:
        return "I couldn't find recipes matching your ingredients and restrictions."

    response = "Here are recipes matching your ingredients:\n\n"

    for i, item in enumerate(results, 1):
        response += f"{i}. {item['title']}\n"
        response += f"   Matched: {', '.join(item['matched_ingredients'])}\n"
        response += f"   Missing: {', '.join(item['missing_ingredients']) or 'None'}\n"
        response += f"   Additional ingredients: {', '.join(item['additional_ingredients'])}\n"
        response += f"   Match score: {item['match_score']}\n\n"

    return response


def substitution_response(target_ingredient):
    results = get_substitution_recommendations(target_ingredient)

    if not results:
        return (
            f"I couldn't find verified substitution evidence "
            f"for {target_ingredient} in the recipe dataset."
        )

    response = f"Substitution suggestions for {target_ingredient}:\n\n"

    for item in results[:5]:
        response += f"Recipe: {item['recipe_title']}\n"
        response += f"Substitute: {item['substitute_ingredient']}\n"

        quantity = item["suggested_quantity"]

        if quantity:
            response += (
                f"Suggested quantity: "
                f"{quantity['quantity']} {quantity['unit']}\n"
            )
        else:
            response += "Suggested quantity: Not specified\n"

        response += f"Evidence: {item['evidence']}\n"
        response += f"Recipe ID: {item['recipe_id']}\n\n"

    return response


def answer_recipe_question(message):
    recipe_id = chat_history["selected_recipe_id"]

    if not recipe_id:
        return "Please select a recipe first."

    recipe = recipes.get(recipe_id)

    if not recipe:
        return "Recipe details are unavailable."

    instructions = recipe.get("instructions") or ""

    if isinstance(instructions, list):
        instructions = " ".join(instructions)

    instructions = str(instructions)

    if any(word in message for word in ["how long", "time", "minutes", "hours"]):
        sentences = re.split(r'(?<=[.!?])\s+', instructions)
        relevant = [
            sentence for sentence in sentences
            if re.search(r'\b(bake|cook|roast)\b', sentence, re.I)
            and re.search(r'\b\d+\s*(?:minutes?|hours?|mins?|hrs?)\b', sentence, re.I)
        ]

        if relevant:
            return "\n".join(relevant)

        return "I couldn't find a specific cooking time in this recipe."

    return "I couldn't find an answer in the selected recipe instructions."


def get_total_recipe_count():
    return len(recipes)


def get_common_ingredients(limit=10):
    ingredient_counts = Counter()

    for recipe in recipes.values():
        ingredients = recipe.get("normalized_ingredients") or []
        ingredient_counts.update(
            ingredient for ingredient in ingredients if ingredient
        )

    return ingredient_counts.most_common(limit)


def get_ingredient_statistics():
    ingredient_counts = [
        len(recipe.get("normalized_ingredients") or [])
        for recipe in recipes.values()
    ]

    return {
        "average": round(statistics.mean(ingredient_counts), 2),
        "median": statistics.median(ingredient_counts)
    }


def get_recipes_by_instruction_steps(limit=10):
    results = []

    for recipe in recipes.values():
        instructions = recipe.get("instructions") or ""

        if isinstance(instructions, list):
            steps = instructions
        else:
            steps = re.split(r'(?<=[.!?])\s+', str(instructions))

        steps = [step.strip() for step in steps if step.strip()]

        results.append({
            "id": recipe["id"],
            "title": recipe.get("title"),
            "step_count": len(steps)
        })

    results.sort(
        key=lambda item: item["step_count"],
        reverse=True
    )

    return results[:limit]


def statistics_response(query):
    query = query.lower()

    if "average" in query or "median" in query:
        stats = get_ingredient_statistics()

        return (
            f"Average ingredients per recipe: {stats['average']}\n"
            f"Median ingredients per recipe: {stats['median']}"
        )

    elif any(phrase in query for phrase in [
        "common ingredients",
        "most used ingredients",
        "commonly used ingredients"
    ]):
        results = get_common_ingredients(10)

        response = "Top 10 most common ingredients:\n"

        for ingredient, count in results:
            response += f"- {ingredient}: {count}\n"

        return response

    elif "total recipes" in query or "how many recipes" in query:
        return f"Total recipes in the dataset: {get_total_recipe_count()}"

    elif "most steps" in query or "instruction steps" in query:
        results = get_recipes_by_instruction_steps(10)

        response = "Recipes ranked by instruction count:\n"

        for item in results:
            response += f"- {item['title']}: {item['step_count']} steps\n"

        return response

    return "I couldn't identify the statistics you requested."


def recipe_specific_substitution(target, replacement):
    recipe_id = chat_history.get("selected_recipe_id")

    if not recipe_id:
        return "Please select a recipe first."

    recipe = recipes.get(recipe_id)

    if not recipe:
        return "Selected recipe not found."

    ingredients = recipe.get("ingredients") or []

    if not any(target.lower() in item.lower() for item in ingredients):
        return f"{target} was not found in {recipe['title']}."

    evidence = find_substitutions(target)

    relevant = [
        item for item in evidence
        if replacement.lower() in item["evidence"].lower()
    ]

    response = f"Recipe: {recipe['title']}\n"
    response += f"Original ingredient: {target}\n"
    response += f"Requested replacement: {replacement}\n\n"

    if relevant:
        response += "Dataset evidence:\n"
        for item in relevant[:3]:
            response += f"- {item['evidence']}\n"
            response += f"Recipe ID: {item['id']}\n"
        response += "\nThis evidence comes from other recipes; "
        response += "it does not confirm the substitution will work in this recipe."
    else:
        response += (
            f"I couldn't find dataset evidence confirming "
            f"{replacement} as a substitute for {target}. "
            "I can't confirm that it will work in this recipe."
        )

    return response
