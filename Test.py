from DecisionTree import DecisionTree

dataCSV = "ModLense.csv"
targetColumn = "contact-lenses"
missing_letter = "?"


def read_csv_with_headers(data_csv, targetName, missingLetter):
    data = []
    with open(data_csv) as f:
        dat = f.readlines()
        columns = dat[0].split(",")
        target = -1
        for x in range(0,len(columns)):
            if targetName == columns[x].strip():
                target = x
        if target == -1:
            return None
        for x in range(1,len(dat)):
            row = dat[x].split(",")
            datarow = {}
            features = {}
            for y in range(0,len(row)):
                if y == target:
                    datarow["label"] = row[y].strip()
                else:
                    val = row[y].strip()
                    if not missingLetter==val:
                        features[columns[y].strip()] = val
            datarow["features"] = features
            data.append(datarow)
    return {"data": data}


dataset = read_csv_with_headers(dataCSV, targetColumn, missing_letter)
dt = DecisionTree()
tree = dt.train_decision_tree(dataset, 0.05)
pred = dt.predict(tree, dataset["data"][1])
print(pred)

agaricusCSV = "agaricus-lepiota.data"
targetName = "class"

mushroom = read_csv_with_headers(agaricusCSV, targetName, missing_letter)
tree2 = dt.train_decision_tree(mushroom,0.5)
pred2 = dt.predict(tree2, mushroom["data"][1])
print(pred2)