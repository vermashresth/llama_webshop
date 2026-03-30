"""
Single source of truth for WebShop LLM experiment goals and product load.

Matches the notebook: synthetic goals from attributes (human_goals=False),
then deterministic shuffle with seed 233 so index i is the same in the
notebook and in Flask session ids fixed_<i>_<run>.
"""
import random
import itertools
from decimal import Decimal
import json
import re
from tqdm import tqdm
from collections import defaultdict
# from web_agent_site.engine.engine import load_products
# from web_agent_site.engine.goal import get_goals
from web_agent_site.utils import DEBUG_PROD_SIZE, DEFAULT_FILE_PATH
from web_agent_site.utils import (
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)
EXPERIMENT_GOALS_SHUFFLE_SEED = 233
PRICE_RANGE = [10.0 * i for i in range(1, 100)]

def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products

def load_products(filepath, num_products=None, human_goals=True):
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(filepath) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    all_reviews = dict()
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    skip_wrong_asins = 0
    skip_no_matching_attr = 0
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            skip_wrong_asins += 1
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']

        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        # if asin in attributes and 'attributes' in attributes[asin]:
        #     products[i]['Attributes'] = attributes[asin]['attributes']
        # else:
        #     products[i]['Attributes'] = ['DUMMY_ATTR']
        # products[i]['instruction_text'] = \
        #     attributes[asin].get('instruction', None)
        # products[i]['instruction_attributes'] = \
        #     attributes[asin].get('instruction_attributes', None)

        # without color, size, price, availability
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
            else:
                skip_no_matching_attr += 1
        else:
            products[i]['instruction_text'] = \
                attributes[asin].get('instruction', None)

            products[i]['instruction_attributes'] = \
                attributes[asin].get('instruction_attributes', None)

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])
    print(f'Skipped {skip_wrong_asins} wrong ASINs.')
    print(f'Skipped {skip_no_matching_attr} no matching attributes.')

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins

def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices

def load_webshop_products_and_goals(shuffle_seed=EXPERIMENT_GOALS_SHUFFLE_SEED):
    """
    Load products and shuffled synthetic goals for the experiment stack.

    Returns:
        all_products, product_item_dict, product_prices, attribute_to_asins, goals
    """
    random.seed(shuffle_seed)
    all_products, product_item_dict, product_prices, attribute_to_asins = load_products(
        filepath=DEFAULT_FILE_PATH,
        num_products=DEBUG_PROD_SIZE,
        human_goals=False,
    )
    goals = get_goals(all_products, product_prices, human_goals=False)
    random.shuffle(goals)
    return all_products, product_item_dict, product_prices, attribute_to_asins, goals

def get_goals(all_products, product_prices, human_goals=True):
    if human_goals:
        return get_human_goals(all_products, product_prices)
    else:
        return get_synthetic_goals(all_products, product_prices)
    
def get_human_goals(all_products, product_prices):
    goals = []
    cnt_atts = defaultdict(int)
    cnt = 0
    for item in all_products:
        asin = item['asin']
        if 'instructions' not in item: continue
        for product in item['instructions']:
            attributes = product['instruction_attributes']
            if len(attributes) == 0: 
                cnt += 1
                continue

            if product_prices is not None:
                price = product_prices[asin]
                price_range = [p for p in PRICE_RANGE if p > price][:4]
                if len(price_range) >= 2:
                    _, price_upper = sorted(random.sample(price_range, 2))
                    price_text = \
                        f', and price lower than {price_upper:.2f} dollars'
                else:
                    price_upper = 1000000
                    price_text = ''
            else:
                price_upper = 1000000

            goals.append({
                'asin': asin,
                'category': item['category'],
                'query': item['query'],
                'name': item['name'],
                'product_category': item['product_category'],
                'instruction_text': product['instruction'].strip('.') + price_text,
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': product['instruction_options'],
            })
            for att in attributes:
                cnt_atts[att] += 1
            # goals += product_goals
    for goal in goals:
        goal['weight'] = 1
    print(cnt, 'skipped')
    return goals


def get_synthetic_goals(all_products, product_prices):
    goals = []
    cnt_atts = defaultdict(int)
    for product in all_products:
        if ('instruction_text' not in product or 
            product['instruction_text'] is None):
            continue
        product_goals = []        
        asin = product['asin']
        attributes = product['instruction_attributes']
        assert len(attributes) > 0

        if product_prices is not None:
            price = product_prices[asin]
            price_range = [p for p in PRICE_RANGE if p > price][:4]
            if len(price_range) >= 2:
                _, price_upper = sorted(random.sample(price_range, 2))
                price_text = \
                    f', and price lower than {price_upper:.2f} dollars'
            else:
                price_upper = 1000000
                price_text = ''
        else:
            price_upper = 1000000
            price_text = ''

        instruction_text = product['instruction_text']

        options = product['options']
        option_names = sorted(options)
        combinations = list(itertools.product(
            *(options[option_name] for option_name in option_names)
        ))
        for combination in combinations:
            goal_options = dict()
            for i, o in enumerate(combination):
#                option_text.append(f'{option_names[i]}: {o}')
                goal_options[option_names[i]] = o
            option_text = ', and '.join([
                f'{k}: {v}' for k, v in goal_options.items()
            ])
            option_text = ' with ' + option_text if option_text else ''
            product_goals.append({
                'asin': asin,
                'category': product['category'],
                'query': product['query'],
                'name': product['name'],
                'product_category': product['product_category'],
                'instruction_text': f'{instruction_text}{option_text}{price_text}',
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': goal_options,
                'name': product['Title'],
            })
            for att in attributes:
                cnt_atts[att] += 1
        goals += product_goals
    for goal in goals:
        goal['weight'] = sum(1. / cnt_atts[att] for att in goal['attributes']) / len(goal['attributes'])
    return goals

