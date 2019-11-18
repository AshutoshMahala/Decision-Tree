from collections import defaultdict
import functools
import math

class DecisionTree:

    def __init__(self):
        pass

    def class_distribution(self, dataset):
        dist = defaultdict(lambda: 0.0)
        for row in dataset:
            dist[row["label"]]+= 1.0
        return dist

    def normalize(self, x):
        sum = functools.reduce(lambda a,b: a+b, x.values())
        if sum<=0.0:
            sum = 1.0
        for y  in x.keys():
            x[y] = 1.0*x[y]/sum
        return x

    def p_logp(self, p):
        if (p<=0.0 or p>=1.0):
            return 0.0
        return p*math.log(p)

    def entropy(self, distribution):
        frequencies = self.normalize(distribution)
        h = 0.0
        for frequency in frequencies.values():
            h-= self.p_logp(frequency)
        return h

    def information_gain(self, h0, splits):
        size = 0.0
        for split in splits.values():
            size += len(split)
        sum = 0.0
        for split in splits.values():
            dist = self.class_distribution(split)
            split_size = functools.reduce(lambda a, b: a + b, dist.values())
            sum += (1.0*split_size/size)*self.entropy(dist)
        return h0-sum

    def split_with_category(self,data, fname):
        splits = defaultdict(lambda: [])
        for row in data:
            if fname in row["features"]:
                splits[row["features"][fname]].append(row)
            else:
                splits["None?"].append(row)
        return splits

    def label_distribution_by_category(self, data, fname):
        distribution = defaultdict(lambda: defaultdict(lambda: 0.0))
        for row in data:
            if fname in row["features"]:
                distribution[row["features"][fname]][row["label"]] += 1.0
            else:
                distribution["None?"][row["label"]]+=1.0
        return distribution

    def find_best_categorical_split(self, data, h0):
        max_ig = -1.0
        max_feature = ""
        features = data[0]["features"].keys()
        for feature in features:
            splits = self.split_with_category(data,feature)
            ig = self.information_gain(h0, splits)
            if max_ig < ig:
                max_ig = ig
                max_feature = feature
        return [max_ig, max_feature]

    def build_categorical_tree(self,data, h0, minsize):
        distribution = self.class_distribution(data)
        tree = {}
        if len(distribution)<=0:
            return None
        if len(distribution)==1:
            freq = self.normalize(distribution)
            tree["isleaf"] = True
            tree["result"] = freq
            return tree
        if len(data)<=minsize:
            frequencies = self.normalize(distribution)
            tree["isleaf"] = True
            max_freq = -1.0
            for key in frequencies.keys():
                if max_freq< frequencies[key]:
                    max_freq = frequencies[key]
                    tree["result"] = {key: frequencies[key]}
            return tree
        ig, feature = self.find_best_categorical_split(data,h0)
        splits = self.split_with_category(data,feature)
        for key in splits.keys():
            hc = self.entropy(self.class_distribution(splits[key]))
            child = self.build_categorical_tree(splits[key],hc, minsize)
            if not child is None:
                tree["isleaf"] = False
                tree["feature"] = feature
                tree[key] = child
        return tree

    def train_decision_tree(self,dataset, eta):
        h0 = self.entropy(self.class_distribution(dataset["data"]))
        minsize = eta*len(dataset["data"])
        return self.build_categorical_tree(dataset["data"], h0, minsize)

    def predict(self, tree, row):
        if tree['isleaf']:
            k =list(tree["result"].keys())[0]
            return {"prediction": k, "confidance": tree["result"][k]}
        else:
            if tree["feature"] in row["features"].keys():
                return self.predict(tree[row["features"][tree["feature"]]], row)
            else:
                return self.predict(tree["None?"], row)