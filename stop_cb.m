function stop_cb(hObj,~)
global handles
handles.run=0;
guidata(hObj,handles)
end