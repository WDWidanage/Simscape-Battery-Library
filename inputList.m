classdef inputList < int32
    enumeration
        curr (0)
        vol (1)
        pow (2)
    end
    methods(Static)
        function map = displayText()
            map = containers.Map;
            map('curr') = 'Current';
            map('vol') = 'Voltage';
            map('pow') = 'Power';
        end
    end
end