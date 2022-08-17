from hyperopt import hp

config = {
    'domain': 'laptop6', 
    'device': 'cuda'
}
bert_mapper = {
    'laptop8': 'activebus/BERT-DK_laptop',
    'laptop6': 'activebus/BERT-DK_laptop',
    'restaurant5': 'activebus/BERT-DK_rest',
    'restaurant3': 'activebus/BERT-DK_rest'
}
path_mapper = {
    'laptop8': './datasets/laptop8',
    'laptop6': './datasets/laptop6',
    'restaurant5': './datasets/restaurant5',
    'restaurant3': './datasets/restaurant3'
}
aspect_category_mapper = {
    'laptop8': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'laptop6': ['support', 'display', 'battery', 'company', 'mouse-keyboard', 'software'],
    'restaurant5': ['ambience', 'drinks', 'food', 'location','service'],
    'restaurant3': ['food', 'place', 'service']
}
aspect_seed_mapper = {
    'laptop8': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"}
    },
    'laptop6': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse-keyboard': {"mouse", "touch", "track", "button", "pad", "keyboard", "key", "space", "type", "keys"},
        'software': {"software", "programs", "applications", "itunes", "photo", "os", "windows", "ios", "mac", "system", "linux"},
    },
    'restaurant5': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "flavourful"},
        'drinks': {"drinks", "wine", "beer", "cola", "water"},
        'location': {"location", "place", "view", "neigborhood", "street", "surroundings"},
        'ambience': {"ambience", "atmosphere", "seating", "decoration", "environment", "decor"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    },
    'restaurant3': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration", "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    }
}
sentiment_category_mapper = {
    'laptop8': ['negative', 'positive'],
    'laptop6': ['negative', 'positive'],
    'restaurant5': ['negative', 'positive'],
    'restaurant3': ['negative', 'positive']
}
sentiment_seed_mapper = {
    'laptop8': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware", "virus", "junk", "crap", "cramped", "cramp"}
    },
    'laptop6': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware", "virus", "junk", "crap", "cramped", "cramp"}
    },
    'restaurant5': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast", "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross", "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"}
    },
    'restaurant3': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast", "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross", "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"}
    }
}

# for weights in loss function
class_categories_dist = {
      "laptop8":     [111, 113, 6240],
      "laptop6":     [109, 112, 4627],
      "restaurant5": [358, 119, 2903],
      "restaurant3": [345, 117, 1566]
    }

#only for CASC
aspect_categories_dist = {
      "laptop8":     [32, 30, 39, 24, 38, 21, 16, 24],
      "laptop6":     [32, 39, 24, 38, 44, 44],
      "restaurant5": [54, 35, 239, 11, 138],
      "restaurant3": [259, 65, 138]
    }
sentiment_categories_dist = {
      "laptop8":     [111,113],
      "laptop6":     [109, 112],
      "restaurant5": [358, 119],
      "restaurant3": [345, 117]
}
aspect_category_mapper = {
    'laptop8': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'laptop6': ['support', 'display', 'battery', 'company', 'mouse-keyboard', 'software'],
    'restaurant5': ['ambience', 'drinks', 'food', 'location','service'],
    'restaurant3': ['food', 'place', 'service']
}

hyperpar_classifier_mapper = {
    'laptop8': {'batch_size': 60.0, 'epochs': 15, 'lr': 9.29e-06},
    'laptop6': {'batch_size': 60.0, 'epochs': 15, 'lr': 9.30e-06},
    'restaurant5': {'batch_size': 64.0, 'epochs': 15, 'lr': 2.59e-06},
    'restaurant3': {'batch_size': 36.0, 'epochs': 15, 'lr': 1.03e-05}
}

hyperpar_labeler_mapper = {
    'laptop8': {"threshold": 0.75, "threshold_na": -0.4, "threshold_attn": 0.05},
    'laptop6': {"threshold": 0.6, "threshold_na": -0.3, "threshold_attn": 0.05},
    'restaurant5': {"threshold": 0.75, "threshold_na": -0.4, "threshold_attn": 0.05},
    'restaurant3': {"threshold": 0.5, "threshold_na": -0.5, "threshold_attn": 0.05}
}
M = {
    'laptop8': 150,
    'laptop6': 150,
    'restaurant5': 100,
    'restaurant3': 100
}
K_1 = 10
K_2 = 30
lambda_threshold = 0.5
batch_size = 32
validation_data_size = 100
learning_rate = 1e-5
epochs = 15

#hyperparam search spaces
bayes_parameters_LDC = {'threshold_attn' : hp.quniform('threshold_attn', 0.04, 0.10, 0.01),
                    'threshold_na' : hp.quniform('threshold_na', -0.60, -0.25, 0.05),
                    'threshold': hp.quniform('threshold', 0.25, 0.75, 0.05) #hp.choice('batch_size',[64])
                    }

bayes_parameters_class = {'lr' : hp.uniform('lr', 1e-7, 1e-4),
                    'epochs' : 9,
                    'batch_size': hp.quniform('batch_size', 8, 64, 4)
                    } 
                    
RANDOM_STATE = 0
NUM_EVALS = 15