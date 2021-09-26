classdef eventList < int32
    enumeration
        timeEvnt (0)
        currEvnt (1)
        volEvnt (2)
        socEvnt (3)
        tempEvnt (4)
    end
    methods(Static)
        function map = displayText()
            map = containers.Map;
            map('currEvnt') = 'Current';
            map('volEvnt') = 'Voltage';
            map('socEvnt') = 'SoC';
        end
    end
end