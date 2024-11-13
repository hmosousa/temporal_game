from src.data import load_qtimelines

trainset = load_qtimelines("train", augment=False, use_cache=False)
trainset.push_to_hub("hugosousa/QTimelines", split="train")

validset = load_qtimelines("valid", augment=False, use_cache=False)
validset.push_to_hub("hugosousa/QTimelines", split="valid")

testset = load_qtimelines("test", augment=False, use_cache=False)
testset.push_to_hub("hugosousa/QTimelines", split="test")
