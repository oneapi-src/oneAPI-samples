// Code snippet provided by Thibault Hennequin
// https://gist.github.com/thennequin/64b4b996ec990c6ddc13a48c6a0ba68c

#include "imgui.h"

/* Declaration */
bool BeginPiePopup( const char* pName, int iMouseButton = 0 );
void EndPiePopup();

bool PieMenuItem( const char* pName, bool bEnabled = true );
bool BeginPieMenu( const char* pName, bool bEnabled = true );
void EndPieMenu();


#if 0
/* Example */
std::string sText;

if( ImGui::IsWindowHovered() && ImGui::IsMouseClicked( 1 ) )
{
  ImGui::OpenPopup( "PieMenu" );
}

if( BeginPiePopup( "PieMenu", 1 ) )
{
  if( PieMenuItem( "Test1" ) ) sText = "Test1";
  if( PieMenuItem( "Test2" ) )
  {
    sText = "Test2";
    new MyImwWindow3();
  }
  if( PieMenuItem( "Test3", false ) ) sText = "Test3";
  if( BeginPieMenu( "Sub" ) )
  {
    if( BeginPieMenu( "Sub sub\nmenu" ) )
    {
      if( PieMenuItem( "SubSub" ) ) sText = "SubSub";
      if( PieMenuItem( "SubSub2" ) ) sText = "SubSub2";
      EndPieMenu();
    }
    if( PieMenuItem( "TestSub" ) ) sText = "TestSub";
    if( PieMenuItem( "TestSub2" ) ) sText = "TestSub2";
    EndPieMenu();
  }
  if( BeginPieMenu( "Sub2" ) )
  {
    if( PieMenuItem( "TestSub" ) ) sText = "TestSub";
    if( BeginPieMenu( "Sub sub\nmenu" ) )
    {
      if( PieMenuItem( "SubSub" ) ) sText = "SubSub";
      if( PieMenuItem( "SubSub2" ) ) sText = "SubSub2";
      EndPieMenu();
    }
    if( PieMenuItem( "TestSub2" ) ) sText = "TestSub2";
    EndPieMenu();
  }

  EndPiePopup();
}
#endif
