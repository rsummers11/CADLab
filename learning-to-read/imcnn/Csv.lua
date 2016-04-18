-- csv.lua
-- class to manage comma separate value file + two directly-usable functions
-- various function to manage csv files
-- these CSV files all have a comma delimiter and use " as the quote character

-- ref:
-- http://www.lua.org/pil/20.4.html
-- http://www.torch.ch/manual/torch/utility#torchclass

require 'torch'

-- example: write csv file
if false then
   local separator = ','  -- optional; use if not a comma
   local csv = Csv("filepath", "w", separator)
   csv:write({"a","b","c"}) -- write header
   for i = 1,done do
      csv:write({a[i], b[i], c[i]}) -- write each data row
   end
   csv:close()
end

-- example: read csv file one record at a time
if false then
   local separator = ',' -- optional; use if not a comma
   local csv = Csv("filepath", "r")
   local header = csv:read() -- header is an array of strings
   while true do
      local data = csv:read()
      if not data then break end
      -- process data, an array of strings
   end
   csv:close() --may not work, need to test?
end

-- example: read entire file at once (should be faster than line by line)
if false then
   local csv = Csv("filepath", "r")
   local alllines = csv:readall()
   -- alllines[i] is an array of strings
   csv:close()
end

do
   -- create class Csv
   local Csv = torch.class("Csv")

   -- initializer
   function Csv:__init(filepath, mode, char)
      local msg = nil
      self.file, msg = io.open(filepath, mode)
      self.separator = char or ','
      if not self.file then error(msg) end
   end

   -- close underlying file
   function Csv:close()
      io.close(self.file)
   end

   -- return iterator that reads all the remaining lines
   function Csv:lines()
      return self.file:lines()
   end

   -- return next record from the csv file
   -- return nill if at end of file
   function Csv:read() 
      local line = self.file:read()
      if not line then return nil end
      return fromcsv(line, self.separator)
   end

   -- return all records as an array
   -- each element of the array is an array of strings
   -- should be faster than reading record by record
   function Csv:readall()
      local all = self.file:read("*all")
      local res = {}
      for line in string.gmatch(all, "([^\n]*)\n") do
   res[#res+1] = fromcsv(line, self.separator)
      end
      return res
   end

   -- write array of strings|numbers to the csv file followed by \n
   -- convert to csv format by inserting commas and quoting where necessary
   -- return nil
   function Csv:write(a)
      res, msg = self.file:write(tocsv(a, self.separator),"\n")
      if res then return end
      error(msg)
   end

   -- the next 3 functions came from
   -- http://www.lua.org/pil/20.4.html
   
   -- static method
   -- convert an array of strings or numbers into a row in a csv file
   function tocsv(t, separator)
      local s = ""
      for _,p in pairs(t) do
   s = s .. separator .. escapeCsv(p, separator)
      end
      return string.sub(s, 2) -- remove first comma
   end
   
   -- private 
   -- enclose commas and quotes between quotes and escape original quotes
   function escapeCsv(s, separator)
      if string.find(s, '["' .. separator .. ']') then
      --if string.find(s, '[,"]') then
   s = '"' .. string.gsub(s, '"', '""') .. '"'
      end
      return s
   end

   -- break record from csv file into array of strings
   function fromcsv(s, separator)
      if not s then error("s is null") end
      s = s .. separator -- end with separator
      local t = {}
      local fieldstart = 1
      repeat
   -- next field is quoted? (starts with "?)
   if string.find(s, '^"', fieldstart) then
      local a, c
      local i = fieldstart
      repeat
         -- find closing quote
         a, i, c = string.find(s, '"("?)', i+1)
      until c ~= '"'  -- quote not followed by quote?
      if not i then error('unmatched "') end
      local f = string.sub(s, fieldstart+1, i-1)
      table.insert(t, (string.gsub(f, '""', '"')))
      fieldstart = string.find(s, separator, i) + 1
   else
      local nexti = string.find(s, separator, fieldstart)
      table.insert(t, string.sub(s, fieldstart, nexti-1))
      fieldstart = nexti + 1
   end
      until fieldstart > string.len(s)
      return t
   end
end
