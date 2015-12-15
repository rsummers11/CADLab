/*
 * Reg Macros - Helper macros based on vtkSetGet.h that makes
 * it easy to creat functions for simple Get and Set functions
 * of class memebers
 */

#ifndef _REG_MACROS_H
#define _REG_MACROS_H

//
// Set built-in type.  Creates member Set"name"() (e.g., SetVisibility());
//
#define SetMacro(name,type) \
virtual void Set##name (type _arg) \
  { \
  if (this->name != _arg) \
    { \
    this->name = _arg; \
    } \
  }

//
// Get built-in type.  Creates member Get"name"() (e.g., GetVisibility());
//
#define GetMacro(name,type) \
virtual type Get##name () { \
  return this->name; \
  }

//
// Create members "name"On() and "name"Off() (e.g., DebugOn() DebugOff()).
// Set method must be defined to use this macro.
//
#define BooleanMacro(name,type) \
  virtual void name##On () { this->Set##name(static_cast<type>(1));}   \
  virtual void name##Off () { this->Set##name(static_cast<type>(0));}

#define SetVector3Macro(name,type) \
virtual void Set##name (type _arg1, type _arg2, type _arg3) \
  { \
  if ((this->name[0] != _arg1)||(this->name[1] != _arg2)||(this->name[2] != _arg3)) \
    { \
    this->name[0] = _arg1; \
    this->name[1] = _arg2; \
    this->name[2] = _arg3; \
    } \
  }; \
virtual void Set##name (type _arg[3]) \
  { \
  this->Set##name (_arg[0], _arg[1], _arg[2]);\
  }

#define GetVector3Macro(name,type) \
virtual type *Get##name () \
{ \
  return this->name; \
} \
virtual void Get##name (type &_arg1, type &_arg2, type &_arg3) \
  { \
    _arg1 = this->name[0]; \
    _arg2 = this->name[1]; \
    _arg3 = this->name[2]; \
  }; \
virtual void Get##name (type _arg[3]) \
  { \
  this->Get##name (_arg[0], _arg[1], _arg[2]);\
  }

#define SetClampMacro(name,type,min,max) \
virtual void Set##name (type _arg) \
  { \
  if (this->name != (_arg<min?min:(_arg>max?max:_arg))) \
    { \
    this->name = (_arg<min?min:(_arg>max?max:_arg)); \
    } \
  } \
virtual type Get##name##MinValue () \
  { \
  return min; \
  } \
virtual type Get##name##MaxValue () \
  { \
  return max; \
  }

#define SetStringMacro(name) \
virtual void Set##name (const char* _arg) \
  { \
  if ( this->name == NULL && _arg == NULL) { return;} \
  if ( this->name && _arg && (!strcmp(this->name,_arg))) { return;} \
  if (this->name) { delete [] this->name; } \
  if (_arg) \
    { \
    size_t n = strlen(_arg) + 1; \
    char *cp1 =  new char[n]; \
    const char *cp2 = (_arg); \
    this->name = cp1; \
    do { *cp1++ = *cp2++; } while ( --n ); \
    } \
   else \
    { \
    this->name = NULL; \
    } \
  }

//
// Get character string.  Creates member Get"name"()
// (e.g., char *GetFilename());
//
#define GetStringMacro(name) \
virtual char* Get##name () { \
  return this->name; \
  }


#endif // _REG_MACROS_H
