with open("ids.txt", "r") as fin:
  line = fin.read()
  start = 0
  finish = 17584
  begin = 1
  end = begin
  while start < finish:
        if start > len(line)-1:
           break
        sub = line[start:start+6]
        if int(sub) == begin:
            begin += 1
            if (begin < end):
                print(f"{begin}:{end} missing")
            end = begin
        else:
            end += 1
        start += 6
  