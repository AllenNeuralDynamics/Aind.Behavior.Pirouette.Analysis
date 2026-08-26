import pickle

with open(
    r"C:\Users\brandon.pratt\Desktop\data\body-kinematics\units\good_units_100hrs_v2.pkl",
    "rb",
) as f:
    data = pickle.load(f)

print(f"Number of Units = {len(data)}")
print(data.keys())

# Peek at the first key's contents
first_key = list(data.keys())[0]
print(f"\nFirst key: {first_key}")
print(type(data[first_key]))
print(data[first_key])
print(data[first_key].keys())
print(f"Depth: {data[first_key][list(data[first_key].keys())[2]]}")
print(f"Amp: {data[first_key][list(data[first_key].keys())[1]]}")
print(f"Channels: {data[first_key][list(data[first_key].keys())[5]]}")
print(f"Positions: {data[first_key][list(data[first_key].keys())[6]]}")
