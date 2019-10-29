function result = findcommon (input1, input2)

result=[];

if isempty(input1)==false & isempty(input2)==false
    
    for i=1:length(input2)
    
        result=[result input1(find(input1==input2(i)))];
    
    end

end
end

%{
function result = findcommon (input1, input2,value)

result=[];

if isempty(input1)==false & isempty(input2)==false
    
    for i=1:length(input2)
    
        if value==true
            
            result=[result input(find(input1==input2(i)))];
            
        end
    
    end

end
end
%}