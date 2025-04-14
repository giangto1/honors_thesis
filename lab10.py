def count_es(string):
    total = 0
    if len(string) == 0:
        return 0
    elif len(string) == 1 and string == 'e':
        return 1
    else:
        # if string[-1] == 'e':
        #     total += 1
        return count_es(string[0])  + count_es(string[1:])


    return total

def min_es(string_list):
    fewest = string_list[0]
    j = 1
    while j < len(string_list):
        if count_es(string_list[j]) < count_es(fewest):
            fewest = string_list[j]
        j += 1
    return fewest

if __name__ == '__main__':
    print(count_es("samwise gamgee")) #ent
    # print(min_es(["isildur", "legolas"])) #isildur
    # print(min_es(["Gandalf"])) #Gandalf
    # print(min_es(["TREEBEARD", "Celeborn"])) #TREEBEARD
