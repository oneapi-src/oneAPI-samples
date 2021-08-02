
//D3D11
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <DirectXMath.h>
#include <DirectXColors.h>

//D3D11 TK

#include "SimpleMath.h"
#include "VertexTypes.h"
#include "PrimitiveBatch.h"
#include "Effects.h"
#include "CommonStates.h"
#include "ScreenGrab.h"
#include "DirectXHelpers.h"
#include "Model.h"
#include "GeometricPrimitive.h"



// stl
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>

//Debug
#include <iostream>


//Windows
#include <wrl/client.h>
#include "wincodec.h"


//Namespaces and convenience aliasing

using Microsoft::WRL::ComPtr;
using VertexType = DirectX::VertexPositionColor;

//Exception from tutorial

namespace DX
{
	inline void ThrowIfFailed(HRESULT hr)
	{
		if (FAILED(hr))
		{
			// Set a breakpoint on this line to catch DirectX API errors
			throw std::exception();
		}
	}
}

int main(int argc, const char* argv[])
{
	int ret = 0;

	//Context and device Setup variables
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> context;
	//derived functionality containers
	ComPtr<ID3D11Device1> d3dDevice;
	ComPtr<ID3D11DeviceContext1> d3dContext;

	ComPtr<ID3D11RenderTargetView>  renderTargetView;
	ComPtr<ID3D11DepthStencilView>  depthStencilView;
	ComPtr<ID3D11InputLayout> inputLayout;
	ComPtr<ID3D11Texture2D>  textureRenderTarget;

	std::unique_ptr<DirectX::CommonStates> states;
	std::unique_ptr<DirectX::BasicEffect> effect;

	std::unique_ptr< DirectX::PrimitiveBatch<VertexType >> batch;

	//Needed for COM capabilities or else exceptions
	DX::ThrowIfFailed(CoInitialize(nullptr));

	std::cout << "Running lean hello world program" << std::endl;

	std::cout << "Initialize" << std::endl;
	int width = 1024;
	int height = 768;
	int outputWidth = (std::max)(width, 1);
	int outputHeight = (std::max)(height, 1);


	std::cout << "Device" << std::endl;
	static const D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_2,
		D3D_FEATURE_LEVEL_9_1,
	};
	UINT creationFlags = 0;
	D3D_FEATURE_LEVEL deviceLevel;

	
	DX::ThrowIfFailed(D3D11CreateDevice(
		nullptr,                            // specify nullptr to use the default adapter
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		creationFlags,
		featureLevels,
		static_cast<UINT>(std::size(featureLevels)),
		D3D11_SDK_VERSION,
		device.ReleaseAndGetAddressOf(),    // returns the Direct3D device created
		&deviceLevel,                    // returns feature level of device created
		context.ReleaseAndGetAddressOf()    // returns the device immediate context
	));

	device.As(&d3dDevice);
	context.As(&d3dContext);

	states = std::make_unique<DirectX::CommonStates>(d3dDevice.Get());

	effect = std::make_unique<DirectX::BasicEffect>(d3dDevice.Get());
	effect->SetVertexColorEnabled(true);

	DX::ThrowIfFailed(
		DirectX::CreateInputLayoutFromEffect<VertexType>(d3dDevice.Get(), effect.get(),
			inputLayout.ReleaseAndGetAddressOf()
			));

	batch = std::make_unique<DirectX::PrimitiveBatch<VertexType>>(d3dContext.Get());



	std::cout << "Resources" << std::endl;
	ID3D11RenderTargetView* nullViews[] = { nullptr };
	d3dContext->OMSetRenderTargets(static_cast<UINT>(std::size(nullViews)), nullViews, nullptr);
	renderTargetView.Reset();
	depthStencilView.Reset();
	d3dContext->Flush();

	//Texture setup to later feed render target
	const UINT textureBufferWidth = 1024U;
	const UINT textureBufferHeight = 768U;
	D3D11_TEXTURE2D_DESC desc;
	std::memset(&desc, 0, sizeof(desc));
	desc.Width = textureBufferWidth;
	desc.Height = textureBufferHeight;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BindFlags = D3D11_BIND_RENDER_TARGET;
	desc.CPUAccessFlags = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;


	DX::ThrowIfFailed(d3dDevice->CreateTexture2D(&desc, nullptr, textureRenderTarget.GetAddressOf()));


	const DXGI_FORMAT depthBufferFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	std::memset(&rtvDesc, 0, sizeof(rtvDesc));
	rtvDesc.Format = desc.Format;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvDesc.Texture2D.MipSlice = 0;

	DX::ThrowIfFailed(d3dDevice->CreateRenderTargetView(textureRenderTarget.Get(), &rtvDesc, renderTargetView.ReleaseAndGetAddressOf()));

	CD3D11_TEXTURE2D_DESC depthStencilDesc(depthBufferFormat, textureBufferWidth, textureBufferHeight, 1, 1, D3D11_BIND_DEPTH_STENCIL);
	ComPtr<ID3D11Texture2D> depthStencil;
	DX::ThrowIfFailed(d3dDevice->CreateTexture2D(&depthStencilDesc, nullptr, depthStencil.GetAddressOf()));

	CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc(D3D11_DSV_DIMENSION_TEXTURE2D);
	DX::ThrowIfFailed(d3dDevice->CreateDepthStencilView(depthStencil.Get(), &depthStencilViewDesc, depthStencilView.ReleaseAndGetAddressOf()));


	std::cout << "Render" << std::endl;

	// camera
	DirectX::SimpleMath::Vector3 cam_pos{ 0.f, 0.f, 0.f };
	DirectX::SimpleMath::Vector3 cam_up{ 0.f, 1.f, 0.f };
	DirectX::SimpleMath::Vector3 cam_view{ 0.1f, 0.f, 1.f };
	using DirectX::SimpleMath::Matrix;
	float aspect = outputWidth / (float) outputHeight;
	float fov = DirectX::XM_PI / 3.f;
	DirectX::SimpleMath::Matrix world, view, projection;
	world = DirectX::SimpleMath::Matrix::Identity;
	view = DirectX::SimpleMath::Matrix::CreateLookAt(cam_pos, cam_view, cam_up);

	//Note that the far plane in the OSPRay render operation assumes infinite far plane.
	projection = Matrix::CreatePerspectiveFieldOfView(fov,
		aspect, 1.e-6f, 10.f);
	//change to grey
	d3dContext->ClearRenderTargetView(renderTargetView.Get(), DirectX::Colors::CornflowerBlue);
	d3dContext->ClearDepthStencilView(depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

	d3dContext->OMSetRenderTargets(1, renderTargetView.GetAddressOf(), depthStencilView.Get());

	// Set the viewport.
	CD3D11_VIEWPORT viewport(0.0f, 0.0f, static_cast<float>(outputWidth), static_cast<float>(outputHeight));

	d3dContext->RSSetViewports(1, &viewport);


	d3dContext->OMSetBlendState(states->Opaque(), nullptr, 0xFFFFFFFF);
	d3dContext->OMSetDepthStencilState(states->DepthNone(), 0);
	d3dContext->RSSetState(states->CullNone());

	//transformations for camera control
	effect->SetProjection(projection);
	effect->SetView(view);
	effect->SetWorld(world);


	effect->Apply(d3dContext.Get());
	
	d3dContext->IASetInputLayout(inputLayout.Get());
	
	//Use batch to draw... 
	batch->Begin();

	
	std::vector<DirectX::VertexPositionColor> vertsColors = { { DirectX::SimpleMath::Vector3{-1.0f, -1.0f, 3.0f}, DirectX::XMFLOAT4{0.9f, 0.5f, 0.5f, 1.0f} } ,
		{ DirectX::SimpleMath::Vector3{-1.0f, 1.0f, 3.0f}, DirectX::XMFLOAT4{0.8f, 0.8f, 0.8f, 1.0f} },
		{ DirectX::SimpleMath::Vector3{1.0f, -1.0f, 3.0f}, DirectX::XMFLOAT4{0.8f, 0.8f, 0.8f, 1.0f} },
		{ DirectX::SimpleMath::Vector3{0.1f, 0.1f, 0.3f}, DirectX::XMFLOAT4{0.5f, 0.9f, 0.5f, 1.0f} }
	};
	std::vector<uint16_t> index = { 0, 1, 2, 1, 2, 3 };
	
	batch->DrawIndexed(D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, 
		index.data(),
		index.size(),
		vertsColors.data(),
		vertsColors.size());
		


	batch->End();
	
	std::cout << "Write" << std::endl;
	DX::ThrowIfFailed(DirectX::SaveWICTextureToFile(d3dContext.Get(), textureRenderTarget.Get(), GUID_ContainerFormatPng, L"hellodxtriangle.png"));

	//COM Tear down
	CoUninitialize();

	return ret;
}
