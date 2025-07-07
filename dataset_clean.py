import json

with open("raw_data.json", "r") as f:
    data = json.load(f)

output_row= dict()
final_list = []
for entry in data:
    conversation = entry["conversation"]
    for i in range(0,len(conversation)+1,2): 
        output_row = {"instruction": "", "output": ""} 
        for j in range(0,i,2):
            output_row["instruction"] = output_row["instruction"] + output_row["output"] + conversation[j]["role"] + ": " + conversation[j]["text"] + "\n"
            output_row["output"] = conversation[j+1]["role"] + ": " + conversation[j+1]["text"]
        final_list.append(output_row)
    #break


#remove {"instruction": "", "output": ""}  from final_list
final_list = [x for x in final_list if x != {"instruction": "", "output": ""}]

for i in final_list:
    print(i)
    print()


#print(output_row)

with open("train.jsonl", "w") as f:
    for i in range(6000):
        print("Writing into train.jsonl, line"+str(i))
        f.write(json.dumps(final_list[i]) + "\n")

with open("test.jsonl", "w") as f:
    for i in range(2000):
        print("Writing into test.jsonl, line"+str(i))
        f.write(json.dumps(final_list[6000+i]) + "\n")  


